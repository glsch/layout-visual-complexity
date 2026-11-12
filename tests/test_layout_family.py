"""The layout adaptation of Kyle–Davidson on synthetic pages with known answers."""

import math

import numpy as np
import pytest
from PIL import Image

from layout_visual_complexity.features import layout
from layout_visual_complexity.features.contract import aggregate
from layout_visual_complexity.features.layout import Layout, LayoutSettings
from layout_visual_complexity.features.page import Page, TextLine, Zone


def box(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def block(x0: float, x1: float, y: float, height: float = 16) -> TextLine:
    return TextLine(f"l{y:g}", box(x0, y - height, x1, y), np.array([[x0, y], [x1, y]]), "r1")


def symmetric_page() -> Page:
    """A 1,000 × 1,600 folio with a centred column of lines, each mirrored about the folio's centre."""
    lines = []
    for number in range(10):
        offset = 80 * number
        lines.append(block(200, 800, 300 + offset))
        lines.append(block(200, 800, 1300 + 16 - offset))
    return Page("ms_1r", 1000, 1600, box(0, 0, 1000, 1600), [Zone("r1", "MainZone", box(150, 250, 850, 1350))], lines)


def measure(page: Page, **settings: object) -> dict[str, float]:
    family = Layout(LayoutSettings(flavours=("mass",), **settings))
    return aggregate(family, family.measure(page))


def test_a_mirror_symmetric_page_scores_full_symmetry() -> None:
    values = measure(symmetric_page(), resolution=200)
    assert values["Layout__Symmetry__left_right_mass"] == pytest.approx(1)
    assert values["Layout__Symmetry__top_bottom_mass"] == pytest.approx(1)
    assert values["Layout__Balance__text_block_mass"] == pytest.approx(0, abs=1e-9)
    assert values["Layout__Balance__folio_mass"] == pytest.approx(0, abs=0.01)


def test_lines_off_to_one_side_are_less_symmetric_and_off_balance() -> None:
    page = symmetric_page()
    page.lines = [block(100, 300 + 40 * (n % 5), 300 + 80 * n) for n in range(12)]
    values = measure(page)
    assert values["Layout__Symmetry__left_right_mass"] < 0.9
    assert values["Layout__Balance__folio_mass"] > 0.1


def test_evenly_spaced_lines_give_the_vertical_autocorrelations_peak_at_their_spacing() -> None:
    raster = np.zeros((120, 80))
    raster[2:110:8, 10:70] = 1.0  # a line every 8 cells
    inside = np.ones_like(raster, dtype=bool)
    correlation = layout.autocorrelation(raster, inside)
    height, lag = layout.autocorrelation_peak(correlation[:, 0])
    assert lag == 8 and height > 0.5
    _, across = layout.autocorrelation_peak(correlation[0, :])
    assert across is None  # one column: the horizontal autocorrelation only falls


def test_two_columns_give_a_horizontal_peak_at_their_distance() -> None:
    raster = np.zeros((100, 100))
    raster[10:90, 10:40] = 1.0
    raster[10:90, 55:85] = 1.0
    correlation = layout.autocorrelation(raster, np.ones_like(raster, dtype=bool))
    _, lag = layout.autocorrelation_peak(correlation[0, :])
    assert lag == 45


def test_one_zone_type_gives_zero_entropy() -> None:
    zones = [Zone("r1", "MainZone", box(0, 0, 100, 100)), Zone("r2", "MainZone", box(200, 0, 300, 100))]
    assert layout.zone_type_entropy(zones) == 0
    two = [*zones, Zone("r3", "MarginTextZone", box(0, 200, 200, 300))]
    assert layout.zone_type_entropy(two) == pytest.approx(1)
    assert math.isnan(layout.zone_type_entropy([]))


def test_blank_space_on_a_known_page() -> None:
    inside = np.ones((100, 50), dtype=bool)
    mass = np.zeros((100, 50))
    mass[20:60, 10:40] = 1.0  # 1,200 of 5,000 cells written
    found = layout.blank_space(mass, inside, LayoutSettings(trim=0.0))
    assert found["share"] == pytest.approx(1 - 1200 / 5000)
    assert found["largest_empty"] == pytest.approx(40 * 50 / 5000)  # the band below the block
    assert found["margin_top"] == pytest.approx(0.20) and found["margin_bottom"] == pytest.approx(0.40)
    assert found["margin_left"] == pytest.approx(0.20) and found["margin_right"] == pytest.approx(0.20)


def test_clutter_counts_zones_their_types_and_yolos_line_types() -> None:
    page = symmetric_page()
    page.regions.append(Zone("r2", "DropCapitalZone", box(160, 260, 190, 300)))
    page.regions.append(Zone("r3", "MainZone", box(1100, 0, 1200, 100)))  # off the folio
    page.yolo_lines = [Zone("y1", "DefaultLine", box(200, 284, 800, 300)), Zone("y2", "HeadingLine", box(200, 364,
                       800, 380)), Zone("y3", "DefaultLine", box(200, 444, 800, 460))]  # fmt: skip
    values = measure(page)
    assert values["Layout__Clutter__zones"] == 2
    assert values["Layout__Clutter__zone_types"] == 2
    assert values["Layout__Clutter__line_types"] == 2


def test_the_ink_layer_needs_the_image_and_finds_its_dark_lines(tmp_path) -> None:
    pixels = np.full((400, 250, 3), 230, np.uint8)
    for y in range(60, 340, 40):
        pixels[y : y + 8, 40:210] = 30
    path = tmp_path / "1r.png"
    Image.fromarray(pixels).save(path)
    page = Page("ms_1r", 500, 800, box(0, 0, 500, 800), image_path=path)
    family = Layout(LayoutSettings(flavours=("ink",), resolution=100))
    assert family.needs == {"image", "folio_crop"}
    assert family.columns() == [c for c in family.columns() if c.endswith("_ink")]
    values = aggregate(family, family.measure(page))
    assert values["Layout__Symmetry__left_right_ink"] == pytest.approx(1, abs=0.02)
    assert values["Layout__Regularity__vertical_ink"] > 0.5


def test_flavours_are_checked() -> None:
    with pytest.raises(ValueError, match="flavours"):
        Layout(LayoutSettings(flavours=("paint",)))
    assert len(Layout().columns()) == 22
