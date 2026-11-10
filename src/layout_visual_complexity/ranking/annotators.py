"""Annotators: spellings folded and merged into persons, and persons replaced by pseudonyms.

Names never leave this module's callers' inputs: what's written out is a pseudonym, `annotator_001`, numbered by
first appearance.
"""

import csv
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path

PSEUDONYM = "annotator_{:0{width}d}"


def fold_spelling(spelling: str) -> str:
    """A spelling without accents, case, spaces or punctuation: `Jan Hajič` and `jan_hajic` give `janhajic`."""
    decomposed = unicodedata.normalize("NFKD", spelling)
    plain = "".join(character for character in decomposed if not unicodedata.combining(character))
    return "".join(character for character in plain.casefold() if character.isalnum())


def read_merge_table(path: Path) -> dict[str, str]:
    """The merge table's `spelling` → `person` columns, by folded spelling. Other columns are ignored.

    Raises when one folded spelling is given two persons.
    """
    merges: dict[str, str] = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        missing = {"spelling", "person"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: the merge table has no column {', '.join(sorted(missing))}")
        for number, row in enumerate(reader, 2):
            spelling, person = (row["spelling"] or "").strip(), (row["person"] or "").strip()
            if not spelling or not person:
                raise ValueError(f"{path} line {number}: a spelling or person is empty")
            folded = fold_spelling(spelling)
            if merges.setdefault(folded, person) != person:
                raise ValueError(f"{path} line {number}: a folded spelling is given two persons")
    return merges


def person_key(spelling: str, merges: Mapping[str, str]) -> tuple[str, str]:
    """Who wrote a spelling: the merge table's person, or else the folded spelling, a person of its own."""
    folded = fold_spelling(spelling)
    return ("table", merges[folded]) if folded in merges else ("folded", folded)


def pseudonyms(persons_in_order: Iterable[object], minimum_width: int = 3) -> dict[object, str]:
    """`annotator_001`, `annotator_002`, … in the order persons first appear."""
    order = list(dict.fromkeys(persons_in_order))
    width = max(minimum_width, len(str(len(order))))
    return {person: PSEUDONYM.format(number, width=width) for number, person in enumerate(order, 1)}
