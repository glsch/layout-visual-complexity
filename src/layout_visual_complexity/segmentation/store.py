"""The layout store: Parquet, in the full image's pixels.

`layout/<document>.parquet` holds one row per element of a page: its folio (the quadrilateral, its method as its
class), YOLO's folio box when one was used, YOLO's regions and lines, and Kraken's lines, with `page`, `kind`
(`folio`, `folio_box`, `region`, `yolo_line`, `line`), `class`, `polygon` (a list of `[x, y]` points), `baseline`
(Kraken's lines only), `confidence` (YOLO's), `region` (the id of a Kraken line's region) and `id` (unique within
the page: `folio`, `folio_box`, `r1`…, `y1`…, `l1`…). `pages.parquet` holds one row per page.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from layout_visual_complexity.segmentation.records import PageRecord, Point

STAGES = ("fetch", "folio_box", "folio", "regions", "line_maps", "lines")
POINTS = pa.list_(pa.list_(pa.int32()))
LAYOUT_SCHEMA = pa.schema(
    [
        ("page", pa.string()),
        ("kind", pa.dictionary(pa.int8(), pa.string())),
        ("class", pa.dictionary(pa.int16(), pa.string())),
        ("polygon", POINTS),
        ("baseline", POINTS),
        ("confidence", pa.float32()),
        ("region", pa.string()),
        ("id", pa.string()),
    ]
)
PAGE_COLUMNS = [
    "page",
    "document",
    "label",
    "source",
    "canvas",
    "image_service",
    "width",
    "height",
    "fetched_width",
    "fetched_height",
    "folio_method",
    "folio_iou",
    "regions",
    "yolo_lines",
    "lines",
    *[f"{stage}_seconds" for stage in STAGES],
    "status",
    "settings",
]


def scaler(record: PageRecord) -> tuple[float, float]:
    """From the image the stages worked on to the full image's pixels."""
    if record.width and record.height and record.full_width and record.full_height:
        return record.full_width / record.width, record.full_height / record.height
    return 1.0, 1.0


def scaled(points: list[Point], scale: tuple[float, float]) -> list[list[int]]:
    return [[round(x * scale[0]), round(y * scale[1])] for x, y in points]


def box_points(box: tuple[float, float, float, float]) -> list[Point]:
    x0, y0, x1, y1 = box
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def layout_rows(record: PageRecord) -> list[dict[str, Any]]:
    """A page's elements as rows, in the full image's pixels."""
    scale = scaler(record)
    rows: list[dict[str, Any]] = []

    def add(kind: str, cls: str, polygon: list[Point], element: str, **more: Any) -> None:
        row = {"page": record.page, "kind": kind, "class": cls, "polygon": scaled(polygon, scale), "baseline": None}
        rows.append(row | {"confidence": None, "region": None, "id": element} | more)

    if record.folio is not None:
        add("folio", record.folio.method, record.folio.quadrilateral, "folio")
    if record.folio_box is not None:
        add("folio_box", record.folio_box.cls, box_points(record.folio_box.box), "folio_box",
            confidence=record.folio_box.confidence)  # fmt: skip
    for number, region in enumerate(record.regions, start=1):
        add("region", region.cls, box_points(region.box), f"r{number}", confidence=region.confidence)
    for number, line in enumerate(record.yolo_lines, start=1):
        add("yolo_line", line.cls, box_points(line.box), f"y{number}", confidence=line.confidence)
    for number, line in enumerate(record.lines, start=1):
        region = f"r{line.region + 1}" if line.region is not None else None
        add("line", "line", line.boundary, f"l{number}", baseline=scaled(line.baseline, scale), region=region)
    return rows


def write_layout(path: Path, records: list[PageRecord]) -> Path:
    """One document's pages, in their order, as one Parquet file, written whole."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for record in sorted(records, key=lambda r: r.position) for row in layout_rows(record)]
    table = pa.Table.from_pylist(rows, schema=LAYOUT_SCHEMA)
    partial = path.with_name(path.name + ".part")
    pq.write_table(table, partial, compression="zstd", compression_level=9)
    partial.replace(path)
    return path


def read_layout(path: Path) -> pd.DataFrame:
    """A layout file as a table; polygons and baselines as lists of `[x, y]` points."""
    table = pq.read_table(path).to_pandas()
    for column in ("kind", "class"):
        table[column] = table[column].astype(str)
    return table


def page_row(record: PageRecord, status: str, source: dict[str, Any], settings: str) -> dict[str, Any]:
    folio = record.folio
    return {
        "page": record.page,
        "document": record.document,
        "label": source.get("label"),
        "source": source.get("source"),
        "canvas": source.get("canvas"),
        "image_service": source.get("image_service"),
        "width": record.full_width,
        "height": record.full_height,
        "fetched_width": record.width,
        "fetched_height": record.height,
        "folio_method": folio.method if folio else None,
        "folio_iou": folio.iou if folio else None,
        "regions": len(record.regions),
        "yolo_lines": len(record.yolo_lines),
        "lines": len(record.lines),
        **{f"{stage}_seconds": record.seconds.get(stage) for stage in STAGES},
        "status": status,
        "settings": settings,
    }


def write_pages(path: Path, rows: list[dict[str, Any]]) -> Path:
    """The pages table, in the order given."""
    table = pd.DataFrame(rows, columns=PAGE_COLUMNS)
    for column in ("width", "height", "fetched_width", "fetched_height", "regions", "yolo_lines", "lines"):
        table[column] = table[column].astype("Int32")
    for column in ("folio_iou", *[f"{stage}_seconds" for stage in STAGES]):
        table[column] = table[column].astype("Float64")
    partial = Path(path).with_name(Path(path).name + ".part")
    table.to_parquet(partial, index=False, compression="zstd")
    partial.replace(path)
    return Path(path)


def read_pages(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
