"""The anonymised dataset: judgments, images and manuscripts as TSV, built from the app's export.

`judgments.tsv` has one row per judgment; `images.tsv` one per image, with its IIIF canvas once resolved;
`manuscripts.tsv` one per manuscript. Annotators appear only as pseudonyms, sessions only as numbers, times only as
days.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from layout_visual_complexity.ranking.annotators import person_key, pseudonyms
from layout_visual_complexity.ranking.export import CALIBRATION, KINDS, SKIPPED

JUDGMENT_COLUMNS = ["judgment", "session", "annotator", "source", "date", "left", "right", "chosen", "response_ms"]
IMAGE_COLUMNS = [
    "image",
    "manuscript",
    "shelfmark",
    "folio_label",
    "iiif_manifest",
    "canvas",
    "image_service",
    "width",
    "height",
    "category",
]
# The categorised metadata's fields that are kept, under the names they're published with. Left out: free text
# (catalogue text, the evidence and rationale for a category), file names in another naming scheme, and the
# bookkeeping of how the records were made.
MANUSCRIPT_FIELDS = {
    "normalized_shelfmark": "manuscript",
    "shelfmark": "shelfmark",
    "category": "category",
    "category_name": "category_name",
    "secondary_categories": "secondary_categories",
    "category_confidence": "category_confidence",
    "canvas_count": "canvases",
    "title": "title",
    "date": "date",
    "location": "location",
    "material": "material",
    "height_mm": "height_mm",
    "width_mm": "width_mm",
    "repository": "repository",
    "city": "city",
    "iiif_manifest_url": "iiif_manifest",
}
MANUSCRIPT_COLUMNS = [
    "manuscript",
    "shelfmark",
    "category",
    "category_name",
    "secondary_categories",
    "category_confidence",
    "pages",
    "canvases",
    "title",
    "date",
    "location",
    "material",
    "height_mm",
    "width_mm",
    "repository",
    "city",
    "iiif_manifest",
]


@dataclass(frozen=True)
class Rules:
    """Which judgments the dataset keeps. Skipped pairs are always dropped.

    Args:
        exclude_calibration: Drop the calibration judgments, the pair the app showed every annotator.
        exclude_sources: Drop the sessions of these sources, by name.
    """

    exclude_calibration: bool = True
    exclude_sources: list[str] = field(default_factory=list)


@dataclass
class Dataset:
    """The three tables."""

    judgments: pd.DataFrame
    images: pd.DataFrame
    manuscripts: pd.DataFrame


@dataclass
class Judgments:
    """The judgments kept, with the counts behind them."""

    table: pd.DataFrame
    counts: pd.DataFrame
    sessions: int
    persons: int


def numbered(prefix: str, count: int, minimum_width: int) -> list[str]:
    width = max(minimum_width, len(str(count)))
    return [f"{prefix}_{number:0{width}d}" for number in range(1, count + 1)]


def count_by_rule(combinations: pd.DataFrame, kept: pd.Series) -> pd.DataFrame:
    """Combinations by source and kind (rated, skipped, calibration), and how many of each were kept."""
    source = combinations["source"].fillna("")
    rows = []
    for name in sorted(source.unique()):
        here = source == name
        row = {"source": name}
        for kind in KINDS:
            row[kind] = int((here & (combinations["kind"] == kind)).sum())
        row["kept"] = int((here & kept).sum())
        rows.append(row)
    table = pd.DataFrame(rows, columns=["source", *KINDS, "kept"])
    total = {"source": "total", **{column: int(table[column].sum()) for column in [*KINDS, "kept"]}}
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)


def build_judgments(combinations: pd.DataFrame, merges: dict[str, str], rules: Rules) -> Judgments:
    """The judgments the rules keep, with sessions renumbered, annotators pseudonymised and times cut to the day.

    Sessions are numbered by start time, then the app's session id; persons by their first session. Both are
    counted after the rules, so only sessions and persons with a judgment kept get a number.
    """
    kept = combinations["kind"] != SKIPPED
    if rules.exclude_calibration:
        kept &= combinations["kind"] != CALIBRATION
    if rules.exclude_sources:
        kept &= ~combinations["source"].isin(list(rules.exclude_sources))
    counts = count_by_rule(combinations, kept)

    rows = combinations[kept].sort_values(["started", "session_id", "position"], kind="stable")
    session_order = list(dict.fromkeys(rows["session_id"]))
    session_names = dict(zip(session_order, numbered("session", len(session_order), 4), strict=True))
    persons = rows["user"].map(lambda spelling: person_key(spelling, merges))
    names = pseudonyms(persons)

    table = pd.DataFrame(
        {
            "judgment": numbered("judgment", len(rows), 5),
            "session": rows["session_id"].map(session_names).to_numpy(),
            "annotator": persons.map(names).to_numpy(),
            "source": rows["source"].fillna("").to_numpy(),
            "date": rows["started"].dt.strftime("%Y-%m-%d").to_numpy(),
            "left": rows["left"].to_numpy(),
            "right": rows["right"].to_numpy(),
            "chosen": rows["chosen"].to_numpy(),
            "response_ms": rows["response_ms"].to_numpy(),
        }
    )
    if not rules.exclude_calibration:
        table["calibration"] = (rows["kind"] == CALIBRATION).to_numpy()
    return Judgments(table=table, counts=counts, sessions=len(session_order), persons=len(names))


def read_image_mapping(path: Path) -> pd.DataFrame:
    """Images from the mapping of image files to shelfmarks and IIIF manifests, one row per image.

    The image id is the file name without its extension; the manuscript id is the normalised shelfmark.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data.get("images", data)
    rows = []
    for name, record in records.items():
        stem, dot, _ = name.rpartition(".")
        rows.append(
            {
                "image": stem if dot else name,
                "manuscript": record["normalized_shelfmark"],
                "shelfmark": record["shelfmark"],
                "folio_label": str(record["folio_or_page_from_filename"]),
                "iiif_manifest": record["iiif_manifest_url"],
            }
        )
    images = pd.DataFrame(rows).sort_values("image", ignore_index=True)
    duplicated = images["image"].duplicated()
    if duplicated.any():
        raise ValueError(f"{path}: {int(duplicated.sum())} image ids appear twice")
    for column in ["canvas", "image_service"]:
        images[column] = pd.Series([None] * len(images), dtype="object")
    for column in ["width", "height"]:
        images[column] = pd.array([pd.NA] * len(images), dtype="Int64")
    return images


def read_manuscript_metadata(path: Path) -> pd.DataFrame:
    """Manuscripts from the categorised metadata (JSON lines), with the fields that are published."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        row = {}
        for source, name in MANUSCRIPT_FIELDS.items():
            value = record.get(source)
            if isinstance(value, list):
                value = ",".join(str(item) for item in value)
            row[name] = value
        rows.append(row)
    manuscripts = pd.DataFrame(rows)
    for column in INTEGER_COLUMNS:
        manuscripts[column] = pd.to_numeric(manuscripts[column], errors="coerce").round().astype("Int64")
    if manuscripts["category"].isna().any():
        raise ValueError(f"{path}: {int(manuscripts['category'].isna().sum())} manuscripts have no category")
    return manuscripts


INTEGER_COLUMNS = ("canvases", "height_mm", "width_mm")


def read_manuscript_overrides(path: Path) -> pd.DataFrame:
    """Manuscript records corrected by hand: a `manuscript` column, then any of the published columns but `pages`.

    A row replaces the manuscript's values in every column the table has; an empty cell leaves the value missing. A
    `note` column is ignored.
    """
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).drop(columns="note", errors="ignore")
    if "manuscript" not in table.columns:
        raise ValueError(f"{path}: the overrides table has no column manuscript")
    unknown = set(table.columns) - set(MANUSCRIPT_COLUMNS) - {"manuscript"}
    if unknown or "pages" in table.columns:
        raise ValueError(
            f"{path}: columns that can't be overridden: {', '.join(sorted(unknown | {'pages'} & set(table.columns)))}"
        )
    if table["manuscript"].duplicated().any():
        raise ValueError(f"{path}: a manuscript appears twice")
    return table


def apply_manuscript_overrides(
    manuscripts: pd.DataFrame, images: pd.DataFrame, overrides: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The manuscripts with their corrected records, and the images, whose manifest follows their manuscript's when
    the overrides give one."""
    manuscripts, images = manuscripts.copy(), images.copy()
    unknown = set(overrides["manuscript"]) - set(manuscripts["manuscript"])
    if unknown:
        raise ValueError(f"the overrides name {len(unknown)} manuscripts that aren't in the metadata")
    for row in overrides.to_dict("records"):
        where = manuscripts["manuscript"] == row["manuscript"]
        for column, value in row.items():
            if column == "manuscript":
                continue
            value = value.strip() or None
            if column in INTEGER_COLUMNS:
                manuscripts[column] = manuscripts[column].astype("Int64")
                manuscripts.loc[where, column] = pd.NA if value is None else int(float(value))
            elif column == "category_confidence":
                manuscripts.loc[where, column] = float("nan") if value is None else float(value)
            else:
                manuscripts[column] = manuscripts[column].astype(object)
                manuscripts.loc[where, column] = value
        if (row.get("iiif_manifest") or "").strip():
            images.loc[images["manuscript"] == row["manuscript"], "iiif_manifest"] = row["iiif_manifest"].strip()
    return manuscripts, images


def build_dataset(judgments: pd.DataFrame, images: pd.DataFrame, manuscripts: pd.DataFrame) -> Dataset:
    """The three tables, checked against each other: every judged image is listed, every image's manuscript too."""
    listed = set(images["image"])
    unknown = (set(judgments["left"]) | set(judgments["right"])) - listed
    if unknown:
        raise ValueError(f"{len(unknown)} judged images aren't in the image list")
    missing = set(images["manuscript"]) - set(manuscripts["manuscript"])
    if missing:
        raise ValueError(f"{len(missing)} manuscripts of the image list have no metadata")
    manuscripts = manuscripts[manuscripts["manuscript"].isin(set(images["manuscript"]))].copy()
    pages = images.groupby("manuscript").size()
    manuscripts["pages"] = manuscripts["manuscript"].map(pages).astype(int)
    manuscripts = manuscripts.sort_values(["category", "manuscript"], ignore_index=True)
    images = images.copy()
    images["category"] = images["manuscript"].map(manuscripts.set_index("manuscript")["category"])
    return Dataset(
        judgments=judgments,
        images=images[IMAGE_COLUMNS],
        manuscripts=manuscripts[MANUSCRIPT_COLUMNS],
    )


def write_dataset(dataset: Dataset, folder: Path, readme: str) -> list[Path]:
    """The tables as TSV, and the README, in `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in [
        ("judgments", dataset.judgments),
        ("images", dataset.images),
        ("manuscripts", dataset.manuscripts),
    ]:
        path = folder / f"{name}.tsv"
        table.to_csv(path, sep="\t", index=False, lineterminator="\n")
        written.append(path)
    path = folder / "README.md"
    path.write_text(readme, encoding="utf-8")
    written.append(path)
    return written


def read_dataset(folder: Path) -> Dataset:
    """The tables of a dataset folder. Empty cells read as missing, except `source`, which reads as empty text."""
    folder = Path(folder)
    judgments = pd.read_csv(
        folder / "judgments.tsv",
        sep="\t",
        dtype={"response_ms": "Int64", "source": str},
        keep_default_na=False,
        na_values={"response_ms": [""]},
    )
    images = pd.read_csv(folder / "images.tsv", sep="\t", dtype={"width": "Int64", "height": "Int64"})
    manuscripts = pd.read_csv(folder / "manuscripts.tsv", sep="\t", dtype={"category": str})
    for table in (judgments, images, manuscripts):
        for column in table.columns:
            if table[column].dtype == object or str(table[column].dtype).startswith("str"):
                table[column] = table[column].astype(object)
    return Dataset(judgments=judgments, images=images, manuscripts=manuscripts)


def readme(dataset: Dataset, rules: Rules, counts: pd.DataFrame, notes: list[str] | None = None) -> str:
    """The dataset's README: what each file and column holds, and the rules it was made with."""
    j, i, m = dataset.judgments, dataset.images, dataset.manuscripts
    excluded = ", ".join(rules.exclude_sources) if rules.exclude_sources else "none"
    lines = [
        "# Pairwise judgments of visual complexity",
        "",
        "Annotators were shown two manuscript pages side by side and asked which of the two layouts they found more",
        "visually complex; they answered by clicking it. Each row of `judgments.tsv` is one such answer.",
        "",
        f"- {len(j):,} judgments in {j['session'].nunique():,} sessions by {j['annotator'].nunique():,} annotators;",
        f"- {len(i):,} images from {len(m):,} manuscripts;",
        f"- calibration judgments {'left out' if rules.exclude_calibration else 'kept, and marked'}; skipped pairs "
        f"left out; sources left out: {excluded}.",
        "",
        "## `judgments.tsv`",
        "",
        "| Column | What it holds |",
        "|---|---|",
        "| `judgment` | The judgment's id, numbered in the order the judgments were made within each session. |",
        "| `session` | The session's id, numbered by the sessions' start. A session showed up to 20 pairs. |",
        "| `annotator` | A pseudonym, numbered by each person's first session. Spellings of one person's name were "
        "merged first. |",
        "| `source` | The campaign the session came from, as the app recorded it; empty for its default campaign. |",
        "| `date` | The day the session started (UTC). |",
        "| `left`, `right` | The images shown on the left and on the right (ids from `images.tsv`). |",
        "| `chosen` | The image chosen as the more visually complex. |",
        "| `response_ms` | The time to answer, in milliseconds. |",
    ]
    if "calibration" in j.columns:
        lines.append("| `calibration` | Whether the pair was the calibration pair every annotator was shown. |")
    lines += [
        "",
        "## `images.tsv`",
        "",
        "| Column | What it holds |",
        "|---|---|",
        "| `image` | The image's id. |",
        "| `manuscript` | The manuscript's id (from `manuscripts.tsv`). |",
        "| `shelfmark` | The manuscript's shelfmark. |",
        "| `folio_label` | The folio or page: as the image's file name gives it, or the canvas's label where the file "
        "name gives the canvas's identifier instead. |",
        "| `iiif_manifest` | The manuscript's IIIF manifest. |",
        "| `canvas` | The page's canvas in that manifest; empty where the label didn't match exactly one canvas. |",
        "| `image_service` | The canvas's IIIF image service, from which the page can be fetched at any size. |",
        "| `width`, `height` | The full image's size in pixels, as the manifest gives it. |",
        "| `category` | The manuscript's category (see below). |",
        "",
        "## `manuscripts.tsv`",
        "",
        "| Column | What it holds |",
        "|---|---|",
        "| `manuscript` | The manuscript's id: its shelfmark, lower case, without spaces or punctuation. |",
        "| `shelfmark` | The shelfmark. |",
        "| `category`, `category_name` | The text category: A, handbooks and manuals; B, religious and theological; "
        "C, other scholarly; D, legal; E, secular and literary; F, administrative. |",
        "| `secondary_categories` | Further categories that apply, comma-separated. |",
        "| `category_confidence` | Where categories were assigned automatically, how confident that was (0 to 1). |",
        "| `pages` | The number of its images in the dataset. |",
        "| `canvases` | The number of canvases in its IIIF manifest. |",
        "| `title`, `date`, `location`, `material` | From the catalogue record. |",
        "| `height_mm`, `width_mm` | The leaves' size, in millimetres, from the catalogue record. |",
        "| `repository`, `city` | Where the manuscript is held. |",
        "| `iiif_manifest` | Its IIIF manifest. |",
        "",
        "## How it was made",
        "",
        "| Source | Rated | Skipped | Calibration | Kept |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in counts.itertuples(index=False):
        name = row.source if row.source else "(default campaign)"
        lines.append(f"| {name} | {row.rated:,} | {row.skipped:,} | {row.calibration:,} | {row.kept:,} |")
    for note in notes or []:
        lines += ["", note]
    return "\n".join(lines).rstrip() + "\n"
