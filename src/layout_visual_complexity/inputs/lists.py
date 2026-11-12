"""Lists of sources, and page tables.

A list file names one source per line, optionally followed by a tab and a selection; blank lines and lines starting
with `#` are skipped. A list may also be a TSV with a header: a `source` (or `manifest`) column, and optionally
`pages` (a selection), `canvas` (a canvas id) and `label` (a canvas label); rows of the same source add up.

A page table is a dataset's `images.tsv`: one row per page, with its id (`image`), its document (`manuscript`), its
canvas, image service or URL, and the full image's size. Its pages keep the table's ids.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

from layout_visual_complexity.inputs import InputError

LIST_SOURCE_COLUMNS = ("source", "manifest")
PAGE_TABLE_COLUMNS = {"image", "manuscript"}


@dataclass
class Entry:
    """One source with what to take from it: a selection, and canvases or labels named one by one."""

    source: str
    selection: str = ""
    canvases: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    @property
    def everything(self) -> bool:
        return not (self.selection.strip() or self.canvases or self.labels)


def _resolved(source: str, folder: Path) -> str:
    """A source as given, with a relative path taken from the list's folder."""
    if "://" in source or Path(source).is_absolute():
        return source
    return str((folder / source).resolve())


def _header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig") as stream:
        return [column.strip() for column in stream.readline().rstrip("\n").split("\t")]


def is_page_table(path: Path) -> bool:
    return path.suffix.lower() == ".tsv" and set(_header(path)) >= PAGE_TABLE_COLUMNS


def read_list(path: Path) -> list[Entry]:
    """The entries of a list file, in order; rows naming the same source merge into one entry."""
    path = Path(path)
    header = _header(path)
    source_column = next((column for column in LIST_SOURCE_COLUMNS if column in header), None)
    entries: dict[str, Entry] = {}
    if path.suffix.lower() == ".tsv" and source_column:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for number, row in enumerate(csv.DictReader(stream, delimiter="\t"), start=2):
                source = (row.get(source_column) or "").strip()
                if not source:
                    raise InputError(f"{path}, row {number}: no source")
                entry = entries.setdefault(source, Entry(_resolved(source, path.parent)))
                selection = (row.get("pages") or "").strip()
                if selection:
                    entry.selection = f"{entry.selection},{selection}" if entry.selection else selection
                if (row.get("canvas") or "").strip():
                    entry.canvases.append(row["canvas"].strip())
                elif (row.get("label") or "").strip():
                    entry.labels.append(row["label"].strip())
        return list(entries.values())
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        source, _, selection = line.partition("\t")
        entry = entries.setdefault(source, Entry(_resolved(source, path.parent)))
        if selection.strip():
            entry.selection = f"{entry.selection},{selection.strip()}" if entry.selection else selection.strip()
    return list(entries.values())


def read_page_table(path: Path) -> list[dict[str, str]]:
    """A page table's rows, every cell as text."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = [
            {key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(stream, delimiter="\t")
        ]
    for number, row in enumerate(rows, start=2):
        if not row.get("image") or not row.get("manuscript"):
            raise InputError(f"{path}, row {number}: no image id or manuscript")
        if not (row.get("image_service") or row.get("image_url")):
            raise InputError(f"{path}, row {number}: no image service or image URL")
    return rows
