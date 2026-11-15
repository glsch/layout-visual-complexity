"""The tables and figures of a collection of manuscripts scored by `lvc score`, for `lvc report` and for
`lvc score --profiles`:

- `manuscript_profiles`: each manuscript's profile (`figures/manuscript_profile_<id>`): its pages as a strip coloured
  by score, the scores with their rolling median over nine pages, and its three most and three least complex pages;
- `manuscript_profile_strips`: the same pages as the article first drew them (`figures/manuscript_profile_strip_<id>`):
  the strip above the six pages, labelled complex and plain, with their scores;
- `manuscript_strips`: every manuscript's strip, one row each (`figures/manuscript_strips`);
- `manuscript_overview`: each manuscript's mean score against the interquartile range of its pages' scores, with
  intervals from resampling its pages, coloured by hands, sized by folios (`figures/<collection>_overview`); and
  `manuscript_overview_quadrants`, the article's design, the quadrants named inside;
- `manuscripts_table`: the collection's table, the catalogue's facts beside the scores and the share of pages with
  colour (`tables/<collection>_manuscripts`);
- `manuscript_readings`: the article's readings as numbers, with intervals (`tables/<collection>_readings`).
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import Normalize

from layout_visual_complexity import manuscripts as ms
from layout_visual_complexity.download import DEFAULT_CACHE, Cache, HttpClient, HttpSettings
from layout_visual_complexity.judgment_reports import Written
from layout_visual_complexity.logs import logger
from layout_visual_complexity.outputs import LatexTable, escape, number, save_figure, write_table
from layout_visual_complexity.plots import manuscripts as plots
from layout_visual_complexity.runs import Run, shown_path
from layout_visual_complexity.segmentation.store import read_pages


@dataclass
class Collection:
    """A score run's manuscripts: their pages' scores in each manuscript's order, the pages' image services, the
    colour of each page, the catalogue's facts, the Elo scale, and each manuscript's statistics."""

    scores: pd.DataFrame
    services: pd.Series
    colour: pd.Series
    catalogue: pd.DataFrame
    elo: dict[str, float]
    left_out: list[str]
    statistics: pd.DataFrame
    drawn: dict[str, ms.Resampled]
    norm: Normalize
    order: list[str] = field(default_factory=list)
    min_lines: int = 0
    blank: dict[str, int] = field(default_factory=dict)

    def label(self, manuscript: str) -> str:
        """A manuscript's shelfmark, else its id."""
        if "shelfmark" in self.catalogue.columns and manuscript in self.catalogue.index:
            return str(self.catalogue.at[manuscript, "shelfmark"])
        return manuscript

    def pages(self, manuscript: str) -> pd.DataFrame:
        return self.scores[self.scores["document"] == manuscript]


def run_models(folder: Path) -> dict[str, Any]:
    """The ensemble a score run used, from its configuration."""
    from layout_visual_complexity.scoring import ENSEMBLE, shipped_folder

    configuration = yaml.safe_load((Path(folder) / "config.yaml").read_text(encoding="utf-8")) or {}
    models = Path(configuration["models"]) if configuration.get("models") else shipped_folder()
    return json.loads((models / ENSEMBLE).read_text(encoding="utf-8"))


def load_collection(folder: Path, catalogue: Path | None, resamples: int, seed: int, run: Run | None = None,
                    scores: pd.DataFrame | None = None, described: dict | None = None,
                    min_lines: int = 0) -> Collection:  # fmt: skip
    """A score run's manuscripts, with their catalogue's facts if given; `described` is the ensemble the run used,
    read from its configuration when not given. Pages with fewer than `min_lines` of Kraken's text lines are left
    out (blank or nearly blank leaves)."""
    from layout_visual_complexity.features.extraction import read_feature_table
    from layout_visual_complexity.score import FEATURES, SCORES, read_scores

    folder = Path(folder)
    if run is not None:
        for name in (SCORES, "pages.parquet", FEATURES):
            if (folder / name).is_file():
                run.read(folder / name)
    if scores is None:
        scores = read_scores(folder)
    scores = scores.dropna(subset=[ms.SCORE]).sort_values(["document", "order"], kind="stable")
    pages = read_pages(folder / "pages.parquet").set_index("page")
    blank: dict[str, int] = {}
    if min_lines:
        few = pages["lines"].reindex(scores["page"]).fillna(0).to_numpy() < min_lines
        blank = {str(document): int(count) for document, count in scores.loc[few, "document"].value_counts().items()}
        scores = scores[~few]
    services = pages["image_service"].reindex(scores["page"])
    colour = pd.Series(np.nan, index=scores["page"])
    if (folder / FEATURES).is_file():
        colour = ms.colour_pages(read_feature_table(folder / FEATURES).values).reindex(scores["page"])
    facts = ms.read_catalogue(catalogue) if catalogue is not None else pd.DataFrame()
    if run is not None and catalogue is not None:
        run.read(Path(catalogue))
    statistics, drawn = ms.manuscript_statistics(scores, ms.SCORE, resamples, seed)
    described = described if described is not None else run_models(folder)
    values = scores[ms.SCORE].to_numpy(dtype=float)
    low, high = np.percentile(values, [2, 98])
    documents = list(dict.fromkeys(scores["document"]))
    known = [name for name in ms.shelfmark_order(facts) if name in documents] if len(facts) else []
    order = known + [name for name in documents if name not in known]
    return Collection(scores.reset_index(drop=True), services, colour, facts, described.get("elo", {}),
                      list(described.get("trained_on", {}).get("left_out", [])), statistics, drawn,
                      Normalize(float(low), float(high)), order, min_lines, blank)  # fmt: skip


def scale_sentence(collection: Collection, manuscript: str | None = None) -> str:
    """How a predicted score is made, and which pages count (the manuscript's, or the collection's), for the
    captions."""
    elo = collection.elo
    text = (
        "A page's predicted score is its expected Elo score: the regression of the dataset's Elo scores on the "
        "ensemble's out-of-fold scores, "
        f"{number(elo.get('intercept', math.nan), 0)} + {number(elo.get('slope', math.nan), 0)} $\\times$ $z$, $z$ "
        "being the page's score under the ensemble of the geometry and the ViT rankers (equal weights)."
    )
    if collection.left_out:
        names = ", ".join(collection.label(name) for name in collection.left_out)
        text += f" The rankers were trained without the dataset's pages from {names}, so every score is out of sample."
    return text + pages_sentence(collection, manuscript)


def pages_sentence(collection: Collection, manuscript: str | None = None) -> str:
    """Which pages count, for the captions: the leaves left out, the manuscript's or the collection's."""
    count = sum(collection.blank.values()) if manuscript is None else collection.blank.get(manuscript, 0)
    if not collection.min_lines or not count:
        return ""
    leaves = "leaf" if count == 1 else "leaves"
    whose = "The" if manuscript is None else "Its"
    return (f" {whose} {count:,} {leaves} on which Kraken finds fewer than {collection.min_lines} text lines "
            "(blank or nearly blank) are left out.")  # fmt: skip


def credits(image_credit: str, catalogue_credit: str = "") -> str:
    return "".join(f" {escape(text)}" for text in (catalogue_credit, image_credit) if text)


def shown_pages(collection: Collection, manuscript: str, size: int, client: Any, cache: Any) -> list[plots.ShownPage]:
    """The three most and three least complex pages of a manuscript, with their thumbnails."""
    pages = collection.pages(manuscript).reset_index(drop=True)
    scores = pages.set_index("page")[ms.SCORE]
    most, least = ms.extremes(scores, 3)
    place = {page: index for index, page in enumerate(pages["page"])}
    labels = pages.set_index("page")["label"]
    found = []
    for chosen, is_complex in ((most, True), (least, False)):
        for page in chosen:
            service = collection.services.get(page)
            image = None
            if isinstance(service, str) and service:
                try:
                    image = ms.fetch_thumbnail(service, size, client, cache)
                except Exception as error:  # a thumbnail that can't be had is left blank
                    logger.warning("%s: no thumbnail: %s", page, error)
            found.append(plots.ShownPage(place[page], page, str(labels[page]), float(scores[page]), image, is_complex))
    return found


def profile_caption(collection: Collection, manuscript: str, design: str, image_credit: str,
                    catalogue_credit: str) -> str:  # fmt: skip
    pages = collection.pages(manuscript)
    label = collection.label(manuscript)
    facts = collection.catalogue.loc[manuscript] if manuscript in collection.catalogue.index else None
    number_note = f" ({escape(facts['catalogue_number'])})" if facts is not None and "catalogue_number" in facts else ""
    if design == "ours":
        drawn = (
            "(top) each page as a band coloured by its predicted score; (middle) the scores (dots) with their rolling "
            "median over nine pages (line); (bottom) the three most complex pages (1--3, vermillion) and the three "
            "least complex (4--6, blue), each with its folio and score, marked on the strip and among the scores."
        )
    else:
        drawn = (
            "the strip of its pages, each coloured by its predicted score, above its three most complex pages "
            "(vermillion) and its three least complex (blue), each with its folio and score, their places boxed on "
            "the strip."
        )
    return (f"{escape(label)}{number_note}, its {len(pages):,} leaf pages in the manuscript's order: {drawn} "
            f"{scale_sentence(collection, manuscript)} The colour scale runs from {number(collection.norm.vmin, 0)} to "
            f"{number(collection.norm.vmax, 0)}, the 2nd and 98th percentiles of all the collection's pages."
            + credits(image_credit, catalogue_credit))  # fmt: skip


@dataclass(frozen=True)
class CollectionSettings:
    """What the collection's outputs need, from `lvc report`'s or `lvc score`'s settings."""

    score_run: Path
    catalogue: Path | None = None
    profiles: list[str] = field(default_factory=list)
    collection: str = "manuscripts"
    image_credit: str = ""
    catalogue_credit: str = ""
    thumbnail_size: int = 400
    named: list[str] = field(default_factory=list)
    min_lines: int = 0
    bootstrap: int = 1000
    seed: int = 1
    cache: Path = DEFAULT_CACHE
    http: HttpSettings = field(default_factory=HttpSettings)


def profiles(settings: CollectionSettings, collection: Collection, folder: Path, design: str) -> Written:
    """Each chosen manuscript's profile, in our design or the article's."""
    chosen = settings.profiles or collection.order
    unknown = [name for name in chosen if name not in collection.order]
    if unknown:
        raise ValueError(f"no scored pages of {', '.join(unknown)}")
    paths, summary = [], {}
    cache = Cache(Path(settings.cache).expanduser() / "thumbnails", ".jpg")
    with HttpClient(settings.http) as client:
        for manuscript in chosen:
            pages = collection.pages(manuscript)
            values = pages[ms.SCORE].to_numpy(dtype=float)
            shown = shown_pages(collection, manuscript, settings.thumbnail_size, client, cache)
            if design == "ours":
                figure = plots.plot_profile(values, list(pages["label"].astype(str)), shown, collection.norm)
                stem = folder / "figures" / f"manuscript_profile_{manuscript}"
            else:
                figure = plots.plot_profile_strip(values, shown, collection.norm)
                stem = folder / "figures" / f"manuscript_profile_strip_{manuscript}"
            caption = profile_caption(collection, manuscript, design, settings.image_credit, settings.catalogue_credit)
            paths += save_figure(figure, stem, caption, f"fig:{stem.name}")
            summary[manuscript] = [{"page": page.page, "score": page.score, "complex": page.complex} for page in shown]
    return Written(paths, summary)


def strips(settings: CollectionSettings, collection: Collection, folder: Path) -> Written:
    rows = [(collection.label(name), collection.pages(name)[ms.SCORE].to_numpy(dtype=float))
            for name in collection.order]  # fmt: skip
    figure = plots.plot_strips(rows, collection.norm)
    ordered = "in the catalogue's order" if len(collection.catalogue) else "in the run's order"
    caption = (
        f"The {len(rows)} manuscripts' leaf pages, one row each {ordered}, each page a band coloured by its predicted "
        "score, in the manuscript's order from the left; the rows' lengths are the manuscripts' numbers of leaf pages. "
        f"{scale_sentence(collection)}" + credits("", settings.catalogue_credit)
    )
    paths = save_figure(figure, folder / "figures" / "manuscript_strips", caption, "fig:manuscript_strips")
    return Written(paths, {"manuscripts": len(rows)})


def overview_table(collection: Collection) -> pd.DataFrame:
    """The statistics with the catalogue's hands, folios and fine count of texts, and each manuscript's label."""
    table = collection.statistics.reindex(collection.order).copy()
    facts = collection.catalogue.reindex(table.index)
    table["hands"] = facts["hands"] if "hands" in facts.columns else np.nan
    table["folios"] = facts["folios"] if "folios" in facts.columns else table["pages"] / 2
    texts = facts["texts_fine"] if "texts_fine" in facts.columns else pd.Series(np.nan, index=table.index)
    table["label"] = [
        collection.label(name) + (f" ({int(texts[name])})" if pd.notna(texts[name]) else "") for name in table.index
    ]
    return table


def overview(settings: CollectionSettings, collection: Collection, folder: Path, design: str) -> Written:
    table = overview_table(collection)
    figure = plots.plot_overview(table, design)
    stem = folder / "figures" / (f"{settings.collection}_overview" + ("" if design == "ours" else "_quadrants"))
    drawn = (
        "Each manuscript's mean predicted score (horizontal) against the interquartile range of its pages' predicted "
        "scores (vertical): how complex its pages are on average, and how much they vary. Points are coloured by the "
        "number of hands the catalogue identifies and sized by its number of folios; each is labelled with its "
        "shelfmark and, in brackets, the number of texts it holds (counted finely). The dashed lines are the "
        "medians of the manuscripts' means and ranges."
    )
    if design == "ours":
        drawn += (
            " The whiskers are 95\\% intervals from 1,000 resamples of each manuscript's pages. Plain and uniform "
            "manuscripts lie towards the lower left, complex and varied ones towards the upper right; plain but "
            "varied ones towards the upper left, complex but uniform ones towards the lower right."
        ).replace("1,000", f"{settings.bootstrap:,}")
    else:
        drawn += " The quadrants the medians mark are named in their corners."
    caption = f"{drawn} {scale_sentence(collection)}" + credits("", settings.catalogue_credit)
    paths = save_figure(figure, stem, caption, f"fig:{stem.name}")
    return Written(paths, {"manuscripts": len(table)})


def manuscripts_table(settings: CollectionSettings, collection: Collection, folder: Path) -> Written:
    """The collection's table: the catalogue's facts, the pages scored, the mean and interquartile range of their
    predicted scores with intervals, and the share of pages with colour beside the catalogue's colours."""
    facts = collection.catalogue.reindex(collection.order)
    stats = collection.statistics.reindex(collection.order)
    colour = collection.colour.groupby(collection.scores.set_index("page")["document"]).mean()
    table = pd.DataFrame(
        {
            "manuscript": [collection.label(name) for name in collection.order],
            "catalogue_number": facts.get("catalogue_number"),
            "date": [ms.short_date(text) for text in facts.get("date", pd.Series("", index=facts.index))],
            "date_as_catalogued": facts.get("date"),
            "folios": facts.get("folios"),
            "hands": facts.get("hands"),
            "texts_coarse": facts.get("texts_coarse"),
            "texts_fine": facts.get("texts_fine"),
            "languages": [ms.languages(text) for text in facts.get("languages", pd.Series("", index=facts.index))],
            "pages": stats["pages"],
            **{column: stats[column] for column in ("mean", "mean_low", "mean_high", "iqr", "iqr_low", "iqr_high")},
            "colour_share": colour.reindex(collection.order),
            "colours": facts.get("colours"),
        },
        index=collection.order,
    )

    def interval(value: float, low: float, high: float) -> str:
        return f"{number(value, 0)} [{number(low, 0)}, {number(high, 0)}]"

    rows = [
        [escape(row.manuscript), escape(row.catalogue_number or ""), escape(row.date), number(row.folios, 0),
         number(row.hands, 0), number(row.texts_fine, 0), escape(row.languages), number(int(row.pages), 0),
         interval(row.mean, row.mean_low, row.mean_high), interval(row.iqr, row.iqr_low, row.iqr_high),
         number(row.colour_share, 2), number(row.colours, 0)]
        for row in table.itertuples()
    ]  # fmt: skip
    caption = (
        "The case study's manuscripts in the catalogue's order: shelfmark, catalogue number, date (the earliest and "
        "latest year the catalogue names, else its centuries as written), folios, hands, texts (counted finely) and "
        "languages (ISO 639-3) as catalogued; the leaf pages scored; the mean of their predicted scores and their "
        f"interquartile range, each with its 95\\% interval from {settings.bootstrap:,} resamples of the manuscript's "
        "pages; the share of its pages with colour (a rubric line, at least half of whose ink is chromatic, or a "
        "coloured initial, by the colour measures, which no ranker uses); and the number of colours besides plain ink "
        f"the catalogue gives. {scale_sentence(collection)}" + credits("", settings.catalogue_credit)
    )
    latex = LatexTable(
        ["Shelfmark", "No.", "Date", "Folios", "Hands", "Texts", "Languages", "Pages", "Mean", "IQR", "Colour",
         "Colours"],
        rows, caption, f"tab:{settings.collection}_manuscripts",
    )  # fmt: skip
    paths = write_table(table.reset_index(drop=True), folder / "tables" / f"{settings.collection}_manuscripts", latex)
    path = paths[-1]
    path.write_text(path.read_text(encoding="utf-8").replace(r"\centering", "\\centering\n  \\footnotesize", 1),
                    encoding="utf-8")  # fmt: skip
    return Written(paths, {"manuscripts": len(table)})


def readings(settings: CollectionSettings, collection: Collection, folder: Path) -> Written:
    """The article's readings, checked: ρ of hands with the mean and the range, composite against homogeneous,
    the named manuscripts' places by range, and the colour share against the catalogue's colours."""
    colour = collection.colour.groupby(collection.scores.set_index("page")["document"]).mean()
    table = ms.readings(collection.statistics, collection.drawn, collection.catalogue, colour,
                        settings.named, settings.bootstrap, settings.seed)  # fmt: skip
    table["note"] = table.get("note", pd.Series("", index=table.index)).fillna("")
    rows = []
    for row in table.itertuples():
        digits = 3 if abs(row.value) < 10 else 1
        given = f"[{number(row.low, digits)}, {number(row.high, digits)}]"
        rows.append([escape(row.reading), number(row.value, digits), given, number(row.manuscripts, 0)])
    caption = (
        "The readings of the overview, checked against the catalogue: Spearman's $\\rho$ over the manuscripts, with "
        f"95\\% intervals from {settings.bootstrap:,} resamples of the manuscripts; the composite volumes' mean "
        "interquartile range less the homogeneous ones' (the catalogue's distinction; the undecided left out), the "
        "interval from resampling the manuscripts within each group; each named manuscript's place among them all by "
        "interquartile range (1 the most varied), its interval from the resamples of every manuscript's pages."
        + credits("", settings.catalogue_credit)
    )
    latex = LatexTable(["Reading", "Value", "95\\% interval", "Manuscripts"], rows, caption,
                       f"tab:{settings.collection}_readings", align="lccc")  # fmt: skip
    paths = write_table(table, folder / "tables" / f"{settings.collection}_readings", latex)
    return Written(paths, {"readings": table.to_dict("records")})


OUTPUTS = (
    "manuscript_profiles",
    "manuscript_profile_strips",
    "manuscript_strips",
    "manuscript_overview",
    "manuscript_overview_quadrants",
    "manuscripts_table",
    "manuscript_readings",
)


def make(output: str, settings: CollectionSettings, collection: Collection, folder: Path) -> Written:
    """One of the collection's outputs."""
    if output == "manuscript_profiles":
        return profiles(settings, collection, folder, "ours")
    if output == "manuscript_profile_strips":
        return profiles(settings, collection, folder, "article")
    if output == "manuscript_strips":
        return strips(settings, collection, folder)
    if output == "manuscript_overview":
        return overview(settings, collection, folder, "ours")
    if output == "manuscript_overview_quadrants":
        return overview(settings, collection, folder, "article")
    if output == "manuscripts_table":
        return manuscripts_table(settings, collection, folder)
    if output == "manuscript_readings":
        return readings(settings, collection, folder)
    raise ValueError(f"unknown output {output!r}")


def score_run_figures(output: Path, scores: pd.DataFrame, settings: Any, described: dict) -> list[Path]:
    """`lvc score --profiles`: each document's profile and the overview of the documents, into `figures/`."""
    given = CollectionSettings(
        score_run=Path(output),
        image_credit=settings.image_credit,
        thumbnail_size=settings.thumbnail_size,
        cache=Path(settings.cache),
        http=settings.http,
    )
    collection = load_collection(Path(output), None, given.bootstrap, given.seed, scores=scores, described=described)
    paths = profiles(given, collection, Path(output), "ours").paths
    if len(collection.order) > 1:
        paths += overview(given, collection, Path(output), "ours").paths
    logger.info("score: %d figures in %s", sum(path.suffix == ".svg" for path in paths), shown_path(Path(output)))
    return paths
