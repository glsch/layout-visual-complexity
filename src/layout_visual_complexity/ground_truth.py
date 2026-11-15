"""Ground truth for pages: their zones and lines as people drew and transcribed them, read from ALTO files or from a
CATMuS-style `metadata.jsonl`.

A page's zones each have a SegmOnto class and a polygon; its lines a class, a polygon, a baseline, a text (none where
the source gives none) and the zone holding them. Class names lose a SegmOnto subtype after `#` or `:`, so
`MainZone#2` is `MainZone`. Coordinates are in the image's own pixels.

ALTO's classes come from its `Tags`, which each file numbers itself, so they're read per file. A line without a
class is a `DefaultLine`, SegmOnto's default; a block without a class holds lines but isn't a zone. In a
`metadata.jsonl` every image is listed by its `file_name` beside the file, in folders by manuscript; an image listed
twice is read once, as first listed.
"""

import hashlib
import json
import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field, replace
from pathlib import Path

from layout_visual_complexity.inputs.lists import read_list
from layout_visual_complexity.inputs.selection import document_id, page_ids
from layout_visual_complexity.logs import logger

Point = tuple[float, float]
DEFAULT_LINE = "DefaultLine"


@dataclass(frozen=True)
class Zone:
    """A zone: its id in the source, its SegmOnto class (None where the source gives none) and its polygon."""

    id: str
    cls: str | None
    polygon: list[Point]


@dataclass(frozen=True)
class Line:
    """A line: its id, class, polygon, baseline, text (None where the source has no transcriptions), and the zone
    holding it, by id and class."""

    id: str
    cls: str
    polygon: list[Point]
    baseline: list[Point]
    text: str | None = None
    zone: str | None = None
    zone_cls: str | None = None


@dataclass(frozen=True)
class Page:
    """A page's ground truth: its id and document, its image and the image's size, its zones and lines, the file it
    was read from with that file's SHA-256, and, for half of a spread, the part of the image it is (x0, y0, x1, y1,
    in the image's pixels; its own coordinates start at the part's corner)."""

    page: str
    document: str
    image: Path
    width: int
    height: int
    zones: list[Zone] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    source: Path = Path()
    checksum: str = ""
    region: tuple[int, int, int, int] | None = None
    label: str = ""

    @property
    def size(self) -> tuple[int, int]:
        """The page's own width and height: the image's, or its part's."""
        if self.region is None:
            return self.width, self.height
        x0, y0, x1, y1 = self.region
        return x1 - x0, y1 - y0

    @property
    def transcribed(self) -> bool:
        """Whether the page's lines carry texts."""
        return any(line.text is not None for line in self.lines)


def base_class(name: str | None) -> str | None:
    """A SegmOnto class without its subtype: `MainZone#2` and `MainZone:column` are `MainZone`."""
    if name is None:
        return None
    found = re.split(r"[#:]", name.strip(), maxsplit=1)[0].strip()
    return found or None


def numbers(text: str | None) -> list[float]:
    """The numbers in a list of coordinates, whether `x y x y` or `x,y x,y`."""
    return [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", text or "")]


def pairs(values: list[float]) -> list[Point]:
    return [(values[i], values[i + 1]) for i in range(0, len(values) - 1, 2)]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as opened:
        for block in iter(lambda: opened.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ALTO.


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in element if _local(child.tag) == name]


def _polygon(element: ElementTree.Element) -> list[Point]:
    """An element's `Shape/Polygon`, else its box."""
    for shape in _children(element, "Shape"):
        for polygon in _children(shape, "Polygon"):
            found = pairs(numbers(polygon.get("POINTS")))
            if len(found) >= 3:
                return found
    try:
        x, y = float(element.get("HPOS", "")), float(element.get("VPOS", ""))
        w, h = float(element.get("WIDTH", "")), float(element.get("HEIGHT", ""))
    except ValueError:
        return []
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _text(line: ElementTree.Element) -> str | None:
    """A line's text: its `String`s' contents, `SP` as spaces; None without any `String`."""
    parts, found = [], False
    for child in line:
        name = _local(child.tag)
        if name in ("String", "HYP"):
            parts.append(child.get("CONTENT", ""))
            found = True
        elif name == "SP":
            parts.append(" ")
    return "".join(parts) if found else None


def _tag_class(element: ElementTree.Element, tags: dict[str, str]) -> str | None:
    for reference in (element.get("TAGREFS") or "").split():
        if reference in tags:
            return base_class(tags[reference])
    return None


def read_alto(path: Path, document: str | None = None) -> Page:
    """A page's ground truth from an ALTO file, its image beside it (`sourceImageInformation/fileName`, else an image
    of the same stem); its document is its folder's name unless given."""
    path = Path(path)
    root = ElementTree.parse(path).getroot()
    tags = {}
    for element in root.iter():
        if _local(element.tag).endswith("Tag") and element.get("ID") and element.get("LABEL"):
            tags[element.get("ID", "")] = element.get("LABEL", "")
    image = _alto_image(root, path)
    from PIL import Image

    with Image.open(image) as opened:
        width, height = opened.size
    page_element = next((element for element in root.iter() if _local(element.tag) == "Page"), None)
    scale = (1.0, 1.0)
    if page_element is not None and page_element.get("WIDTH") and page_element.get("HEIGHT"):
        declared = float(page_element.get("WIDTH", "0")), float(page_element.get("HEIGHT", "0"))
        if declared[0] > 0 and declared[1] > 0 and (round(declared[0]), round(declared[1])) != (width, height):
            scale = (width / declared[0], height / declared[1])
            logger.info(
                "%s: its page is %g × %g, its image %d × %d; coordinates scaled", path, *declared, width, height
            )

    def scaled(points: list[Point]) -> list[Point]:
        return [(x * scale[0], y * scale[1]) for x, y in points]

    zones, lines = [], []
    for block in root.iter():
        if _local(block.tag) != "TextBlock":
            continue
        block_id = block.get("ID") or f"block{len(zones) + 1}"
        cls = _tag_class(block, tags)
        if cls is not None:
            zones.append(Zone(block_id, cls, scaled(_polygon(block))))
        for line in block.iter():
            if _local(line.tag) != "TextLine":
                continue
            baseline = scaled(pairs(numbers(line.get("BASELINE"))))
            lines.append(
                Line(
                    line.get("ID") or f"line{len(lines) + 1}",
                    _tag_class(line, tags) or DEFAULT_LINE,
                    scaled(_polygon(line)),
                    baseline,
                    _text(line),
                    block_id if cls else None,
                    cls,
                )
            )
    # Lines outside every block.
    inside = {id(line) for block in root.iter() if _local(block.tag) == "TextBlock" for line in block.iter()}
    for line in root.iter():
        if _local(line.tag) == "TextLine" and id(line) not in inside:
            lines.append(
                Line(
                    line.get("ID") or f"line{len(lines) + 1}",
                    _tag_class(line, tags) or DEFAULT_LINE,
                    scaled(_polygon(line)),
                    scaled(pairs(numbers(line.get("BASELINE")))),
                    _text(line),
                )
            )
    folder = document or document_id(path.parent.name)
    stem = image.stem
    return Page(
        page_ids(folder, [stem])[0], folder, image, width, height, zones, lines, path, checksum(path), label=stem
    )


def _alto_image(root: ElementTree.Element, path: Path) -> Path:
    for element in root.iter():
        if _local(element.tag) == "fileName" and (element.text or "").strip():
            named = path.parent / Path(element.text.strip().replace("\\", "/")).name
            if named.is_file():
                return named
    from layout_visual_complexity.inputs.local import is_image

    stem = path.name.split(".")[0]
    for candidate in sorted(path.parent.iterdir()):
        if is_image(candidate) and candidate.name.split(".")[0] == stem:
            return candidate
    raise ValueError(f"{path}: no image beside it")


# CATMuS's metadata.jsonl.


def read_metadata(path: Path, manuscripts: list[str] | None = None) -> list[Page]:
    """The pages of a `metadata.jsonl`, its images beside it in folders by manuscript, in the file's order; only the
    named manuscripts' (their folders) when given. Each `file_name` is read once, as first listed."""
    path = Path(path)
    wanted = set(manuscripts or [])
    pages, seen = [], set()
    found_folders = set()
    file_sum = checksum(path)
    with open(path, encoding="utf-8") as lines:
        for number, text in enumerate(lines, start=1):
            if not text.strip():
                continue
            row = json.loads(text)
            name = str(row["file_name"])
            folder = name.split("/")[0]
            if (wanted and folder not in wanted) or name in seen:
                continue
            seen.add(name)
            found_folders.add(folder)
            pages.append(_metadata_page(row, path, file_sum, number))
    missing = sorted(wanted - found_folders)
    if missing:
        raise ValueError(f"{path}: no images of {', '.join(missing)}")
    return pages


def _metadata_page(row: dict, path: Path, file_sum: str, number: int) -> Page:
    name = str(row["file_name"])
    folder = name.split("/")[0]
    image = path.parent / name
    document = document_id(folder)
    stem = Path(name).stem
    objects = row.get("objects") or {}
    count = len(objects.get("id", []))
    column = {key: list(values) for key, values in objects.items()}
    zones, lines = [], []
    blocks = {}
    for i in range(count):
        if column["type"][i] == "block":
            blocks[str(column["id"][i])] = base_class(column["category"][i])
    for i in range(count):
        polygon = pairs([float(value) for value in column["polygons"][i]])
        identifier, cls = str(column["id"][i]), base_class(column["category"][i])
        if column["type"][i] == "block":
            zones.append(Zone(identifier, cls, polygon))
        else:
            parent = str(column.get("parent", [""] * count)[i] or "") or None
            baseline = pairs([float(value) for value in (column.get("baseline", [[]] * count)[i] or [])])
            lines.append(
                Line(
                    identifier,
                    cls or DEFAULT_LINE,
                    polygon,
                    baseline,
                    None,
                    parent if parent in blocks else None,
                    blocks.get(parent or ""),
                )
            )
    # The file's checksum and the row's number together say what was read for this page.
    identity = hashlib.sha256(f"{file_sum}:{number}".encode()).hexdigest()
    return Page(
        page_ids(document, [stem])[0],
        document,
        image,
        int(row["width"]),
        int(row["height"]),
        zones,
        lines,
        path,
        identity,
        label=stem,
    )


# Sources.


def read_sources(sources: list[str], alto_files: str = "*.xml", manuscripts: list[str] | None = None) -> list[Page]:
    """The pages of ALTO files, folders holding them (at any depth, the files matching `alto_files`), lists of these,
    and `metadata.jsonl` files (only the named manuscripts' images, when given); each page once, by id, as first
    found."""
    found: list[Page] = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            files = sorted(path.rglob(alto_files), key=lambda item: (str(item.parent), item.name))
            found += [read_alto(file) for file in files]
        elif path.suffix == ".jsonl":
            found += read_metadata(path, manuscripts)
        elif path.suffix.lower() == ".xml":
            found.append(read_alto(path))
        elif path.is_file():
            found += read_sources([entry.source for entry in read_list(path)], alto_files, manuscripts)
        else:
            raise ValueError(f"{source}: no such file or folder")
    unique: dict[str, Page] = {}
    for page in found:
        unique.setdefault(page.page, page)
    return list(unique.values())


def shifted(page: Page, region: tuple[int, int, int, int], suffix: str) -> Page:
    """A part of a page as a page of its own: its zones and lines moved to the part's coordinates, its id with
    `suffix`."""
    x0, y0 = region[0], region[1]

    def move(points: list[Point]) -> list[Point]:
        return [(x - x0, y - y0) for x, y in points]

    zones = [replace(zone, polygon=move(zone.polygon)) for zone in page.zones]
    lines = [replace(line, polygon=move(line.polygon), baseline=move(line.baseline)) for line in page.lines]
    return replace(
        page, page=f"{page.page}_{suffix}", zones=zones, lines=lines, region=region, label=f"{page.label}_{suffix}"
    )
