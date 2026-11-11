"""The published dataset: judgments, pages and manuscripts as TSV, with a README, and the checks that hold them
together.

`judgments.tsv` has one row per judgment; `images.tsv` one per page, with its IIIF canvas and a direct link to its
image; `manuscripts.tsv` one per manuscript, with its description. `validate_dataset` defines what a dataset must
satisfy: whoever writes one checks it, and `lvc judgments import` checks it again.
"""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

JUDGMENT_COLUMNS = ["judgment", "session", "annotator", "campaign", "date", "left", "right", "chosen", "response_ms"]
OPTIONAL_JUDGMENT_COLUMNS = ["calibration"]
IMAGE_COLUMNS = [
    "image",
    "manuscript",
    "folio",
    "canvas",
    "image_url",
    "image_service",
    "width",
    "height",
    "work",
    "text",
    "text_language",
]
MANUSCRIPT_COLUMNS = [
    "manuscript",
    "shelfmark",
    "library",
    "city",
    "country",
    "title",
    "date",
    "date_not_before",
    "date_not_after",
    "date_edtf",
    "place_of_origin",
    "material",
    "height_mm",
    "width_mm",
    "languages",
    "texts",
    "category",
    "secondary_categories",
    "pages",
    "canvases",
    "iiif_manifest",
    "catalogue_record",
    "mirabile_id",
    "biblissima_id",
    "census_id",
    "licence",
]
TABLES = {"judgments": JUDGMENT_COLUMNS, "images": IMAGE_COLUMNS, "manuscripts": MANUSCRIPT_COLUMNS}
FILES = ["README.md", "judgments.tsv", "images.tsv", "manuscripts.tsv"]
INTEGERS = {
    "judgments": ["response_ms"],
    "images": ["width", "height"],
    "manuscripts": ["date_not_before", "date_not_after", "height_mm", "width_mm", "texts", "pages", "canvases"],
}
# The forms of EDTF (ISO 8601-2, level 1) the dataset uses: a year, approximate (~), uncertain (?) or both (%); a
# range of years, either end qualified the same ways; a range open at one end.
_YEAR = r"\d{4}[~?%]?"
EDTF = re.compile(rf"^({_YEAR}|{_YEAR}/{_YEAR}|{_YEAR}/\.\.|\.\./{_YEAR})$")
LANGUAGES = re.compile(r"^[a-z]{3}( [a-z]{3})*$")


class DatasetError(ValueError):
    """A dataset that doesn't hold together. The message lists every problem found."""


@dataclass
class Dataset:
    """The three tables."""

    judgments: pd.DataFrame
    images: pd.DataFrame
    manuscripts: pd.DataFrame


def _column_problems(name: str, table: pd.DataFrame) -> list[str]:
    columns, actual = TABLES[name], list(table.columns)
    if name == "judgments" and actual[len(columns) :] == OPTIONAL_JUDGMENT_COLUMNS:
        actual = actual[: len(columns)]
    if actual == columns:
        return []
    missing = [c for c in columns if c not in actual]
    extra = [c for c in actual if c not in columns]
    details = []
    if missing:
        details.append(f"missing {', '.join(missing)}")
    if extra:
        details.append(f"unexpected {', '.join(extra)}")
    return [f"{name}.tsv: the columns aren't as defined ({'; '.join(details) or 'their order differs'})"]


def _blank(values: pd.Series) -> pd.Series:
    return values.isna() | (values.astype(str).str.strip() == "")


def _sample(values: pd.Series | list, count: int = 3) -> str:
    values = list(dict.fromkeys(str(value) for value in values))
    return ", ".join(values[:count]) + (f" and {len(values) - count} more" if len(values) > count else "")


def validate_dataset(dataset: Dataset) -> None:
    """Raise `DatasetError` unless the dataset is as defined: every column in its order, ids unique, every judgment's
    pages listed, every page's manuscript listed, every page with a canvas and an image link, dates in the EDTF forms
    used, languages as ISO 639-3 codes, every manuscript with a category."""
    problems = []
    for name in TABLES:
        problems += _column_problems(name, getattr(dataset, name))
    if problems:
        raise DatasetError("\n".join(problems))
    judgments, images, manuscripts = dataset.judgments, dataset.images, dataset.manuscripts
    for name, table, key in [
        ("judgments", judgments, "judgment"),
        ("images", images, "image"),
        ("manuscripts", manuscripts, "manuscript"),
    ]:
        repeated = table[key][table[key].duplicated()]
        if len(repeated):
            problems.append(f"{name}.tsv: {len(repeated)} ids appear twice ({_sample(repeated)})")
        if _blank(table[key]).any():
            problems.append(f"{name}.tsv: {int(_blank(table[key]).sum())} rows have no {key}")
    pages = set(images["image"])
    for column in ("left", "right", "chosen"):
        unknown = judgments.loc[~judgments[column].isin(pages), column]
        if len(unknown):
            problems.append(
                f"judgments.tsv: {len(unknown)} judgments name a page not in images.tsv in `{column}` "
                f"({_sample(unknown)})"
            )
    other = (judgments["chosen"] != judgments["left"]) & (judgments["chosen"] != judgments["right"])
    if other.any():
        problems.append(f"judgments.tsv: {int(other.sum())} judgments chose neither of their pages")
    itself = judgments["left"] == judgments["right"]
    if itself.any():
        problems.append(f"judgments.tsv: {int(itself.sum())} judgments set a page against itself")
    unknown = images.loc[~images["manuscript"].isin(set(manuscripts["manuscript"])), "manuscript"]
    if len(unknown):
        problems.append(
            f"images.tsv: {len(unknown)} pages name a manuscript not in manuscripts.tsv ({_sample(unknown)})"
        )
    for column in ("canvas", "image_url"):
        if _blank(images[column]).any():
            missing = images.loc[_blank(images[column]), "image"]
            problems.append(f"images.tsv: {len(missing)} pages have no {column} ({_sample(missing)})")
    dates = manuscripts["date_edtf"].fillna("").astype(str).str.strip()
    malformed = manuscripts.loc[(dates != "") & ~dates.str.match(EDTF), "date_edtf"]
    if len(malformed):
        problems.append(f"manuscripts.tsv: {len(malformed)} dates aren't in the EDTF forms used ({_sample(malformed)})")
    languages = manuscripts["languages"].fillna("").astype(str).str.strip()
    malformed = manuscripts.loc[(languages != "") & ~languages.str.match(LANGUAGES), "languages"]
    if len(malformed):
        problems.append(
            f"manuscripts.tsv: {len(malformed)} language lists aren't ISO 639-3 codes ({_sample(malformed)})"
        )
    if _blank(manuscripts["category"]).any():
        missing = manuscripts.loc[_blank(manuscripts["category"]), "manuscript"]
        problems.append(f"manuscripts.tsv: {len(missing)} manuscripts have no category ({_sample(missing)})")
    if problems:
        raise DatasetError("\n".join(problems))


def _read_table(path: Path, name: str) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for column in INTEGERS[name]:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column].replace("", None), errors="coerce").astype("Int64")
    if name == "judgments" and "calibration" in table.columns:
        table["calibration"] = table["calibration"].str.lower().isin(["true", "1", "yes"])
    for column in table.columns:
        if str(table[column].dtype) not in ("Int64", "bool"):
            table[column] = table[column].astype(object)
    return table


def read_dataset(folder: Path) -> Dataset:
    """The tables of a dataset folder. Empty text cells read as empty text, empty numbers as missing."""
    folder = Path(folder)
    missing = [name for name in FILES if not (folder / name).is_file()]
    if missing:
        raise DatasetError(f"{folder}: no {', '.join(missing)}")
    judgments, images, manuscripts = (_read_table(folder / f"{name}.tsv", name) for name in TABLES)
    return Dataset(judgments, images, manuscripts)


def write_dataset(dataset: Dataset, folder: Path, readme: str) -> list[Path]:
    """The README and the three tables in `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = [folder / "README.md"]
    written[0].write_text(readme, encoding="utf-8")
    for name in TABLES:
        path = folder / f"{name}.tsv"
        getattr(dataset, name).to_csv(path, sep="\t", index=False, lineterminator="\n")
        written.append(path)
    return written


def copy_dataset(source: Path, folder: Path) -> list[Path]:
    """The dataset's files copied as they are from `source` into `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in FILES:
        target = folder / name
        shutil.copyfile(Path(source) / name, target)
        copied.append(target)
    return copied
