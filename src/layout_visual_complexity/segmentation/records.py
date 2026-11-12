"""Page records: what the stages read and write, passed between processes as plain data.

Coordinates are in the pixels of the image the stages work on (`width` × `height`); the store scales them to the full
image's (`full_width` × `full_height`).
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Point = tuple[float, float]


@dataclass
class Detection:
    """A box a YOLO model found: its class, corners (x0, y0, x1, y1) and confidence."""

    cls: str
    box: tuple[float, float, float, float]
    confidence: float


@dataclass
class FolioRecord:
    """The leaf: how it was found (`otsu`, `yolo` or `image`), its four corners, and the IoU of YOLO's box with
    Otsu's region."""

    method: str
    quadrilateral: list[Point]
    iou: float | None


@dataclass
class Line:
    """A text line: its baseline and boundary polygon, and the index of the region it's assigned to."""

    baseline: list[Point]
    boundary: list[Point]
    region: int | None = None


@dataclass
class Status:
    state: Literal["done", "failed"] = "done"
    reason: str | None = None


@dataclass
class PageRecord:
    """One page as it goes through the stages."""

    page: str
    document: str
    position: int
    image: str | None = None
    width: int | None = None
    height: int | None = None
    full_width: int | None = None
    full_height: int | None = None
    folio_box: Detection | None = None
    folio: FolioRecord | None = None
    regions: list[Detection] = field(default_factory=list)
    yolo_lines: list[Detection] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    line_map: str | None = None
    status: dict[str, Status] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PageRecord":
        def points(values: list[Any]) -> list[Point]:
            return [(float(x), float(y)) for x, y in values]

        def detection(value: dict[str, Any] | None) -> Detection | None:
            if value is None:
                return None
            return Detection(value["cls"], tuple(value["box"]), value["confidence"])  # type: ignore[arg-type]

        folio = data.get("folio")
        return cls(
            **{key: data.get(key) for key in ("page", "document", "position", "image", "width", "height")},
            full_width=data.get("full_width"),
            full_height=data.get("full_height"),
            folio_box=detection(data.get("folio_box")),
            folio=FolioRecord(folio["method"], points(folio["quadrilateral"]), folio["iou"]) if folio else None,
            regions=[detection(region) for region in data.get("regions", [])],  # type: ignore[misc]
            yolo_lines=[detection(line) for line in data.get("yolo_lines", [])],  # type: ignore[misc]
            lines=[
                Line(points(line["baseline"]), points(line["boundary"]), line.get("region"))
                for line in data.get("lines", [])
            ],
            line_map=data.get("line_map"),
            status={name: Status(**status) for name, status in data.get("status", {}).items()},
            seconds=dict(data.get("seconds", {})),
        )

    @property
    def failed(self) -> list[str]:
        """The stages that failed, in the steps' order."""
        return [name for name, status in self.status.items() if status.state == "failed"]
