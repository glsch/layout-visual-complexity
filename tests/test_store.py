import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from layout_visual_complexity.outputs.alto import StepInfo, page_alto
from layout_visual_complexity.segmentation.lines import assign_lines
from layout_visual_complexity.segmentation.records import Detection, FolioRecord, Line, PageRecord, Status
from layout_visual_complexity.segmentation.store import layout_rows, read_layout, read_pages, write_layout, write_pages

ALTO = "{http://www.loc.gov/standards/alto/ns-v4#}"
DEFAULT_SCHEMA = Path("~/.cache/layout-visual-complexity/schemas/alto-4-4.xsd").expanduser()
XLINK_STUB = b"""<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
    targetNamespace="http://www.w3.org/1999/xlink" xmlns:xlink="http://www.w3.org/1999/xlink">
  <xsd:attributeGroup name="simpleLink">
    <xsd:attribute name="type" type="xsd:string" form="qualified"/>
    <xsd:attribute name="href" type="xsd:anyURI" form="qualified"/>
  </xsd:attributeGroup>
</xsd:schema>"""


def rect(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def record(page: str = "msa_1r", position: int = 1) -> PageRecord:
    """A page fetched at 500×750 from a 1000×1500 image: a folio, two regions, three lines (one outside both)."""
    return PageRecord(
        page=page,
        document="msa",
        position=position,
        width=500,
        height=750,
        full_width=1000,
        full_height=1500,
        folio_box=Detection("Folio", (20.0, 30.0, 480.0, 720.0), 0.93),
        folio=FolioRecord("otsu", [(22.0, 31.5), (478.0, 33.0), (476.0, 718.0), (21.0, 716.0)], 0.97),
        regions=[
            Detection("MainZone", (50.0, 50.0, 400.0, 600.0), 0.88),
            Detection("MarginTextZone", (410.0, 50.0, 470.0, 600.0), 0.51),
        ],
        lines=[
            Line([(60.0, 100.0), (390.0, 100.0)], rect(60, 90, 390, 103)),
            Line([(380.0, 140.0), (460.0, 140.0)], rect(380, 130, 460, 143)),  # mostly in the margin
            Line([(60.0, 700.0), (390.0, 700.0)], rect(60, 690, 390, 703)),  # in no region
        ],
        status={name: Status() for name in ("folio_box", "folio", "regions", "lines")},
    )


def test_lines_are_assigned_to_the_region_they_overlap_most() -> None:
    page = record()
    assign_lines(page.lines, page.regions)
    assert [line.region for line in page.lines] == [0, 1, None]


def test_a_page_fetched_smaller_gives_full_image_coordinates() -> None:
    page = record()
    assign_lines(page.lines, page.regions)
    rows = {row["id"]: row for row in layout_rows(page)}
    assert rows["folio"]["polygon"] == [[44, 63], [956, 66], [952, 1436], [42, 1432]]
    assert rows["folio_box"]["polygon"] == [[40, 60], [960, 60], [960, 1440], [40, 1440]]
    assert rows["r1"]["polygon"][2] == [800, 1200] and rows["r1"]["class"] == "MainZone"
    assert rows["l2"]["baseline"] == [[760, 280], [920, 280]] and rows["l2"]["region"] == "r2"


def test_the_store_is_written_and_read_back_row_for_row(tmp_path: Path) -> None:
    pages = [record("msa_2r", 2), record("msa_1r", 1)]
    for page in pages:
        assign_lines(page.lines, page.regions)
    path = write_layout(tmp_path / "layout" / "msa.parquet", pages)
    table = read_layout(path)
    expected = [row for page in sorted(pages, key=lambda p: p.position) for row in layout_rows(page)]
    assert len(table) == len(expected) == 2 * 7
    for (_, got), want in zip(table.iterrows(), expected, strict=True):
        assert (got["page"], got["kind"], got["class"], got["id"]) == (
            want["page"],
            want["kind"],
            want["class"],
            want["id"],
        )
        assert [list(point) for point in got["polygon"]] == want["polygon"]
        assert (got["region"] if isinstance(got["region"], str) else None) == want["region"]
        if want["confidence"] is None:
            assert got["confidence"] != got["confidence"] or got["confidence"] is None
        else:
            assert got["confidence"] == pytest.approx(want["confidence"], rel=1e-6)
    assert list(table["kind"].unique()) == ["folio", "folio_box", "region", "line"]


def test_the_pages_table(tmp_path: Path) -> None:
    from layout_visual_complexity.segmentation.store import page_row

    page = record()
    page.seconds = {"fetch": 0.5, "lines": 4.0}
    source = {"label": "0001r", "source": "x.tsv", "canvas": "c", "image_service": "s"}
    path = write_pages(tmp_path / "pages.parquet", [page_row(page, "done", source, "key")])
    (row,) = read_pages(path).to_dict("records")
    assert (row["width"], row["fetched_width"], row["folio_method"], row["regions"], row["lines"]) == (
        1000,
        500,
        "otsu",
        2,
        3,
    )
    assert row["lines_seconds"] == 4.0 and row["status"] == "done" and row["label"] == "0001r"


def schema() -> object:
    """The ALTO 4.4 schema, read with lxml from the file `LVC_ALTO_SCHEMA` names (or the package cache's
    `schemas/alto-4-4.xsd`); XLink's schema is a stub."""
    path = os.environ.get("LVC_ALTO_SCHEMA") or (str(DEFAULT_SCHEMA) if DEFAULT_SCHEMA.is_file() else None)
    if not path:
        pytest.skip("set LVC_ALTO_SCHEMA to the ALTO 4.4 schema file to validate against it")
    etree = pytest.importorskip("lxml.etree")

    class Resolver(etree.Resolver):
        def resolve(self, url, public_id, context):  # type: ignore[no-untyped-def]
            if url and url.endswith("xlink.xsd"):
                return self.resolve_string(XLINK_STUB, context)
            return None

    parser = etree.XMLParser(no_network=True)
    parser.resolvers.add(Resolver())
    return etree.XMLSchema(etree.parse(path, parser))


def alto_text() -> str:
    page = record()
    assign_lines(page.lines, page.regions)
    steps = [
        StepInfo("lines", "kraken 7.1.1; LinesStep 1", {"text_direction": "horizontal-lr"}, {"blla.mlmodel": "ab"})
    ]
    return page_alto(page, "0001r", "https://iiif.example.org/i/1/full/full/0/default.jpg", steps)


def test_alto_holds_the_folio_regions_and_lines_in_full_image_pixels() -> None:
    root = ET.fromstring(alto_text())
    page = root.find(f".//{ALTO}Page")
    assert (page.get("WIDTH"), page.get("HEIGHT"), page.get("PRINTED_IMG_NR")) == ("1000", "1500", "0001r")
    tags = {tag.get("ID"): tag.get("LABEL") for tag in root.iter(f"{ALTO}OtherTag")}
    blocks = {
        block.get("ID"): tags[block.get("TAGREFS")] for block in root.iter(f"{ALTO}TextBlock") if block.get("TAGREFS")
    }
    assert blocks == {"r1": "MainZone", "r2": "MarginTextZone"}
    folio = root.find(f".//{ALTO}GraphicalElement")
    assert tags[folio.get("TAGREFS")] == "Folio (otsu)"
    lines = list(root.iter(f"{ALTO}TextLine"))
    assert [line.get("ID") for line in lines] == ["l1", "l2", "l3"]
    assert lines[0].get("BASELINE") == "120 200 780 200"


def test_alto_validates_against_the_alto_4_schema() -> None:
    validator = schema()
    from lxml import etree

    document = etree.fromstring(alto_text().encode())
    assert validator.validate(document), validator.error_log  # type: ignore[attr-defined]
