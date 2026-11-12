"""The real models on one page, end to end. Skipped when `models/` lacks them or the `segment` extra isn't installed."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from layout_visual_complexity.segment import SegmentSettings, run_segment
from layout_visual_complexity.segmentation.regions import FolioBoxSettings, RegionSettings
from layout_visual_complexity.segmentation.store import read_layout, read_pages

MODELS = Path("models")
REGIONS, FOLIO = MODELS / "medieval-yolov11x.pt", MODELS / "folio_yolo.pt"
INSTALLED = all(importlib.util.find_spec(name) for name in ("torch", "ultralytics", "kraken"))

pytestmark = [
    pytest.mark.requires_models,
    pytest.mark.skipif(
        not (INSTALLED and REGIONS.is_file()), reason="needs the segment extra and the regions model in models/"
    ),
]


def written_page(path: Path) -> Path:
    """A leaf on a dark ground, written in two columns of Latin-looking lines, with a heading."""
    image = np.full((1800, 1300, 3), 40, np.uint8)
    cv2.fillPoly(image, [np.array([[70, 50], [1240, 70], [1230, 1760], [60, 1745]], np.int32)], (225, 214, 190))
    rng = np.random.default_rng(3)
    words = ["dominus", "et", "in", "nomine", "patris", "gloria", "sancti", "qui", "regnat", "secula"]
    cv2.putText(image, "INCIPIT LIBER", (230, 210), cv2.FONT_HERSHEY_COMPLEX, 2.0, (40, 40, 160), 4)
    for column in (180, 700):
        for row in range(28):
            text = " ".join(rng.choice(words, 3))
            cv2.putText(image, text, (column, 300 + 48 * row), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.0, (45, 35, 30), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path


@pytest.mark.parametrize("seeded", [False, True])
def test_one_page_end_to_end(tmp_path: Path, seeded: bool) -> None:
    if seeded and not FOLIO.is_file():
        pytest.skip("needs the folio model in models/")
    page = written_page(tmp_path / "ms" / "001r.jpg")
    settings = SegmentSettings(
        sources=[str(page)],
        output=tmp_path / "out",
        cache=tmp_path / "cache",
        jobs=2,
        alto=True,
        folio_box=FolioBoxSettings(model=FOLIO if seeded else None),
        regions=RegionSettings(model=REGIONS),
    )
    summary = run_segment(settings)
    assert summary["done"] == 1, summary["failed"]
    (row,) = read_pages(tmp_path / "out" / "pages.parquet").to_dict("records")
    assert row["folio_method"] == "otsu" and row["lines"] > 20
    assert (row["folio_iou"] is not None and row["folio_iou"] == row["folio_iou"]) == seeded
    layout = read_layout(tmp_path / "out" / "layout" / "ms.parquet")
    assert {"folio", "line"} <= set(layout["kind"]) and ("folio_box" in set(layout["kind"])) == seeded
    xs = [x for polygon in layout["polygon"] for x, _ in polygon]
    assert min(xs) >= 0 and max(xs) <= 1300
    assert (tmp_path / "out" / "alto" / "ms" / "ms_1r.xml").is_file()
    assert not list((tmp_path / "out" / ".maps").glob("*.npz"))


def test_kraken_in_two_halves_gives_what_kraken_gives(tmp_path: Path) -> None:
    from layout_visual_complexity.segmentation.lines import Kraken

    page = str(written_page(tmp_path / "page.jpg"))
    whole = Kraken(None, "cpu", "horizontal-lr").lines(page)
    maps = Kraken(None, "cpu", "horizontal-lr").maps(page, tmp_path / "maps" / "page.npz")
    halves = Kraken(None, "cpu", "horizontal-lr").lines(page, maps)
    assert len(whole) > 20
    assert [(line.baseline, line.boundary) for line in halves] == [(line.baseline, line.boundary) for line in whole]


def test_the_number_of_lines_kraken_keeps_is_a_setting(tmp_path: Path) -> None:
    from layout_visual_complexity.segmentation.lines import Kraken

    page = str(written_page(tmp_path / "page.jpg"))
    assert len(Kraken(None, "cpu", "horizontal-lr", max_lines=10).lines(page)) == 10
    assert len(Kraken(None, "cpu", "horizontal-lr", max_lines=500).lines(page)) > 20
