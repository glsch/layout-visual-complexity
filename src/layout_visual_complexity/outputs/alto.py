"""ALTO 4 output: one file per page, with its folio, regions and lines, in the full image's pixels.

Points are written as one list, `x1 y1 x2 y2 …`, as Kraken's ALTO template writes them and eScriptorium reads them.
Region classes are `OtherTag`s that blocks point to with `TAGREFS`. A region with lines is a `TextBlock`, one without
a `GraphicalElement`; lines outside every region make blocks of their own. The folio is a `GraphicalElement` tagged
with its method.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import layout_visual_complexity
from layout_visual_complexity.segmentation.records import PageRecord
from layout_visual_complexity.segmentation.store import box_points, scaled, scaler

NS = "http://www.loc.gov/standards/alto/ns-v4#"
XSI = "http://www.w3.org/2001/XMLSchema-instance"
SCHEMA_LOCATION = f"{NS} http://www.loc.gov/standards/alto/v4/alto-4-4.xsd"


@dataclass(frozen=True)
class StepInfo:
    """What ALTO's description says of a step: its version, settings and models' checksums."""

    name: str
    version: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)


def element(parent: ET.Element | None, tag: str, **attributes: Any) -> ET.Element:
    """A child in the ALTO namespace, its attributes given as strings; None leaves an attribute out."""
    values = {name: str(value) for name, value in attributes.items() if value is not None}
    if parent is None:
        return ET.Element(f"{{{NS}}}{tag}", values)
    return ET.SubElement(parent, f"{{{NS}}}{tag}", values)


def xml_id(value: str) -> str:
    """An XML ID: letters first, anything an ID can't hold made `_`."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value if value[:1].isalpha() else f"_{value}")


def flat(points: list[list[int]]) -> str:
    return " ".join(f"{x} {y}" for x, y in points)


def bounds(points: list[list[int]]) -> dict[str, int]:
    xs, ys = [x for x, _ in points], [y for _, y in points]
    return {"HPOS": min(xs), "VPOS": min(ys), "WIDTH": max(xs) - min(xs), "HEIGHT": max(ys) - min(ys)}


def describe(description: ET.Element, steps: list[StepInfo]) -> None:
    """One `Processing` per step: its version, models and settings; the software is this package."""
    for step in steps:
        processing = element(description, "Processing", ID=xml_id(f"step_{step.name}"))
        element(processing, "processingCategory").text = "contentGeneration"
        element(processing, "processingStepDescription").text = step.name
        settings = [f"version: {step.version}"] if step.version else []
        settings += [f"model: {name}, SHA-256 {digest}" for name, digest in step.models.items()]
        settings += [f"{key}: {value}" for key, value in step.settings.items()]
        if settings:
            element(processing, "processingStepSettings").text = "; ".join(settings)
        software = element(processing, "processingSoftware")
        element(software, "softwareName").text = "layout-visual-complexity"
        element(software, "softwareVersion").text = layout_visual_complexity.__version__


def page_alto(record: PageRecord, label: str | None, image: str | None, steps: list[StepInfo]) -> str:
    """One page's ALTO file."""
    scale = scaler(record)
    width, height = record.full_width or record.width, record.full_height or record.height
    regions = [(f"r{n}", region) for n, region in enumerate(record.regions, start=1)]
    classes = sorted(
        {region.cls for _, region in regions} | ({f"Folio ({record.folio.method})"} if record.folio else set())
    )
    tags = {cls: f"BT{n}" for n, cls in enumerate(classes, start=1)}

    ET.register_namespace("", NS)
    ET.register_namespace("xsi", XSI)
    alto = element(None, "alto")
    alto.set(f"{{{XSI}}}schemaLocation", SCHEMA_LOCATION)
    description = element(alto, "Description")
    element(description, "MeasurementUnit").text = "pixel"
    if image:
        element(element(description, "sourceImageInformation"), "fileName").text = image
    describe(description, steps)
    if tags:
        tag_list = element(alto, "Tags")
        for cls, tag in tags.items():
            element(tag_list, "OtherTag", ID=tag, LABEL=cls, DESCRIPTION=f"block type {cls}")
    layout = element(alto, "Layout")
    page = element(
        layout, "Page", ID=xml_id(f"page_{record.page}"), PHYSICAL_IMG_NR=record.position, PRINTED_IMG_NR=label,
        WIDTH=width, HEIGHT=height,
    )  # fmt: skip
    space = element(page, "PrintSpace", HPOS=0, VPOS=0, WIDTH=width, HEIGHT=height)
    if record.folio is not None:
        polygon = scaled(record.folio.quadrilateral, scale)
        folio = element(space, "GraphicalElement", ID="folio", TAGREFS=tags[f"Folio ({record.folio.method})"],
                        **bounds(polygon))  # fmt: skip
        element(element(folio, "Shape"), "Polygon", POINTS=flat(polygon))
    members: dict[int | None, list[tuple[int, Any]]] = {}
    for number, line in enumerate(record.lines, start=1):
        members.setdefault(line.region, []).append((number, line))
    for index, (region_id, region) in enumerate(regions):
        polygon = scaled(box_points(region.box), scale)
        tag = "TextBlock" if index in members else "GraphicalElement"
        block = element(space, tag, ID=region_id, TAGREFS=tags[region.cls], **bounds(polygon))
        element(element(block, "Shape"), "Polygon", POINTS=flat(polygon))
        for number, line in members.get(index, []):
            line_element(block, number, line, scale)
    for number, line in members.get(None, []):
        block = element(space, "TextBlock", ID=f"block_l{number}", **bounds(scaled(line.boundary, scale)))
        line_element(block, number, line, scale)
    ET.indent(alto, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(alto, encoding="unicode") + "\n"


def line_element(block: ET.Element, number: int, line: Any, scale: tuple[float, float]) -> None:
    polygon = scaled(line.boundary, scale)
    baseline = scaled(line.baseline, scale)
    text_line = element(block, "TextLine", ID=f"l{number}", BASELINE=flat(baseline), **bounds(polygon))
    element(element(text_line, "Shape"), "Polygon", POINTS=flat(polygon))
    element(text_line, "String", CONTENT="")
