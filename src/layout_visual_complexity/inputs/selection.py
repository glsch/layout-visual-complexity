"""Choosing pages and naming them.

A selection is a comma-separated list of pages and ranges: by index from 1 (`5`, `5-12`, `30-`), or by canvas label
(`10r`, `10r-20v`). An item of digits only is an index; a label of digits only is written `label:12`. A label filter,
a regular expression matched against the folio as page ids write it, keeps only some pages, such as the leaves.

Ids: a document's is its shelfmark or folder name, lower case, without accents, spaces or punctuation (`I.A.29a` →
`ia29a`); a page's is `<document>_<folio>`, the folio being the canvas label in lower case, without spaces or leading
zeros (`0115r` → `115r`), as the published dataset writes them.
"""

import re
import unicodedata
from collections.abc import Sequence

from layout_visual_complexity.inputs import InputError

# Leaves: a number, perhaps a word such as `bis`, `ter` or `quinquies` (or a letter), then recto or verso: `12r`,
# `12bisv`, `12a v`, `136deciesr`. Flyleaves (`Ir`), covers (`FC`) and spines (`SP`) don't pass.
FOLIO_FILTER = r"\d+[a-z]*[rv]"
LABEL_PREFIX = "label:"


def document_id(name: str) -> str:
    """A shelfmark or folder name as a document id: lower case, without accents, spaces or punctuation."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    cleaned = re.sub(r"[^a-z0-9]", "", text)
    if not cleaned:
        raise InputError(f"{name!r} has no letters or digits to make a document id from")
    return cleaned


def page_folio(label: str) -> str:
    """A canvas label as page ids write it: lower case, without spaces or leading zeros (`0115r` → `115r`)."""
    text = re.sub(r"\s+", "", str(label).lower())
    return re.sub(r"^0+(?=\d)", "", text)


def _id_part(folio: str) -> str | None:
    """A folio as it can stand in an id: ASCII, with characters other than letters, digits, `.` and `-` made `_`."""
    text = unicodedata.normalize("NFKD", folio).encode("ascii", "ignore").decode()
    return re.sub(r"[^0-9a-z.-]", "_", text) or None


def page_ids(document: str, labels: Sequence[str | None]) -> list[str]:
    """`<document>_<folio>` for each page, from all of a document's labels, so an id doesn't depend on the selection.

    A page without a label is named by its position (`<document>_p7`); labels that repeat (a front and a back flyleaf,
    both `Ir`) get their position too (`<document>_ir_p3`).
    """
    folios = [_id_part(page_folio(label)) if label else None for label in labels]
    counts: dict[str, int] = {}
    for folio in folios:
        if folio:
            counts[folio] = counts.get(folio, 0) + 1
    ids = []
    for position, folio in enumerate(folios, start=1):
        if not folio:
            ids.append(f"{document}_p{position}")
        elif counts[folio] > 1:
            ids.append(f"{document}_{folio}_p{position}")
        else:
            ids.append(f"{document}_{folio}")
    return ids


def _loose(label: str) -> str:
    """A label compared without case, spaces, leading zeros, or a leading `f.`, `fol.`, `folio` or `p.`."""
    text = re.sub(r"\s+", "", label.lower())
    text = re.sub(r"^(?:folio|fol\.?|f\.?|p\.?)(?=\d)", "", text)
    return re.sub(r"^0+(?=\d)", "", text)


def _find_label(label: str, labels: Sequence[str | None]) -> int:
    """The position (from 1) of the one page with this label."""
    wanted = _loose(label)
    found = [i for i, other in enumerate(labels, start=1) if other and _loose(other) == wanted]
    if not found:
        raise InputError(f"no page has the label {label!r}")
    if len(found) > 1:
        raise InputError(f"the label {label!r} is on several pages ({', '.join(map(str, found))}); use indexes")
    return found[0]


def _position(item: str, labels: Sequence[str | None], count: int) -> int:
    if item.isdigit():
        position = int(item)
        if not 1 <= position <= count:
            raise InputError(f"page {position} is out of range: the document has {count} pages")
        return position
    return _find_label(item.removeprefix(LABEL_PREFIX), labels)


def _range(item: str, labels: Sequence[str | None], count: int) -> list[int]:
    """The positions of one item: a page, or a range between two pages, inclusive; `30-` runs to the end."""
    if not item.startswith(LABEL_PREFIX) and re.fullmatch(r"\d+-", item):
        start, end = int(item[:-1]), count
        if not 1 <= start <= count:
            raise InputError(f"page {start} is out of range: the document has {count} pages")
        return list(range(start, end + 1))
    try:
        return [_position(item, labels, count)]
    except InputError as single:
        # A label may itself hold a hyphen: try each one as the range's divider.
        for index in [i for i, character in enumerate(item) if character == "-"]:
            try:
                start = _position(item[:index].strip(), labels, count)
                end = _position(item[index + 1 :].strip(), labels, count)
            except InputError:
                continue
            if start > end:
                raise InputError(f"the range {item!r} runs backwards (pages {start} to {end})") from None
            return list(range(start, end + 1))
        raise single from None


def select(spec: str, labels: Sequence[str | None]) -> list[int]:
    """The positions (from 1) a selection names, in order, each once; every page when it's empty."""
    if not spec.strip():
        return list(range(1, len(labels) + 1))
    chosen: set[int] = set()
    for item in (part.strip() for part in spec.split(",")):
        if item:
            chosen.update(_range(item, labels, len(labels)))
    return sorted(chosen)


def label_filter(pattern: str | None) -> re.Pattern[str] | None:
    """The label filter as a compiled expression, matched against a whole folio, without case."""
    if not pattern:
        return None
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as error:
        raise InputError(f"the label filter {pattern!r} isn't a regular expression: {error}") from None


def passes(label: str | None, expression: re.Pattern[str] | None) -> bool:
    """Whether a page's label passes the filter; with no filter, every page does, and no label never does."""
    if expression is None:
        return True
    return bool(label) and expression.fullmatch(page_folio(label or "")) is not None
