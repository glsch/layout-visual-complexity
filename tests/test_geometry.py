"""The geometric features on small synthetic layouts with known answers."""

import math

import numpy as np
import pytest

from layout_visual_complexity.features import geometry
from layout_visual_complexity.features.contract import aggregate
from layout_visual_complexity.features.geometry import Geometry, GeometrySettings
from layout_visual_complexity.features.page import Page, TextLine, Zone, shape


def box(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def text_line(
    identifier: str, x0: float, x1: float, y: float, region: str | None = "r1", height: float = 20
) -> TextLine:
    """A horizontal line: its baseline at `y`, its polygon from `height` above it to a quarter of that below."""
    return TextLine(identifier, box(x0, y - height, x1, y + height / 4), np.array([[x0, y], [x1, y]]), region)


def page_with(lines: list[TextLine], zones: list[Zone] | None = None, width: int = 1000, height: int = 1500) -> Page:
    zones = zones if zones is not None else [Zone("r1", "MainZone", box(100, 100, 900, 1400))]
    return Page("ms_1r", width, height, box(0, 0, width, height), zones, lines)


def column(page: Page, name: str) -> float:
    family = Geometry()
    return aggregate(family, family.measure(page))[name]


def test_two_parallel_lines_at_a_known_distance_give_each_interlinear_distance() -> None:
    lines = [text_line("l1", 100, 900, 100), text_line("l2", 100, 900, 150)]
    found = geometry.interlinear_distances(lines, diagonal=1.0, count=50)
    for name in ("min_gap", "mean_gap", "midpoint", "percentile25", "projection_gap"):
        assert found[f"{name}_mean"] == pytest.approx(50)
        assert found[f"{name}_std"] == 0 and found[f"{name}_max_dev"] == 0
    assert found["max_gap_mean"] == pytest.approx(math.hypot(800, 50))


def test_interlinear_distances_are_relative_to_the_folio_diagonal() -> None:
    lines = [text_line(f"l{n}", 100, 900, 100 + 60 * n) for n in range(4)]
    page = page_with(lines)
    assert column(page, "LineFeature__InterlinearDistance__midpoint_median") == pytest.approx(
        60 / math.hypot(1000, 1500)
    )


def test_identical_lines_give_a_dice_of_one() -> None:
    polygons = [shape(box(100, y, 700, y + 30)) for y in (100, 300, 500)]
    found = geometry.line_similarity(polygons)
    assert found["dice_mean"] == pytest.approx(1) and found["iou_median"] == pytest.approx(1)
    assert found["dice_std"] == pytest.approx(0) and found["iou_max_dev"] == pytest.approx(0)


def test_a_written_block_filling_its_rectangle_gives_zero_outside_area() -> None:
    # Lines tiling a block with the folio's proportions (2:3): the union is that rectangle.
    lines = [TextLine(f"l{n}", box(200, 300 + 60 * n, 600, 360 + 60 * n), np.array([[200.0, 350 + 60 * n],
             [600.0, 350 + 60 * n]]), "r1") for n in range(10)]  # fmt: skip
    page = page_with(lines)
    assert column(page, "AreaFeature__WrittenSpaceRectangularity__outside_ratio") == pytest.approx(0, abs=1e-9)
    assert column(page, "AreaFeature__WrittenSpaceRectangularity__perimeter_efficiency") == pytest.approx(1)
    assert column(page, "AreaFeature__WrittenSpaceRatio__convexity") == pytest.approx(1)
    assert column(page, "AreaFeature__WrittenSpaceRatio__folio_ratio") == pytest.approx(400 * 600 / (1000 * 1500))


def test_known_left_offsets_give_the_zigzags_counts_and_shifts() -> None:
    found = geometry.shifts([0.0, 0.1, 0.0, 0.1, 0.1], tolerance=0.005)
    # The first shift starts the count; the next two reverse it; the last step is below the tolerance.
    assert found == {"max_possible": 3, "coord_shift": 2, "zigzag": 2, "zigzag_prop": 2 / 3, "coord_shift_prop": 2 / 3}
    assert math.isnan(geometry.shifts([0.0, 0.1], 0.005)["zigzag"])  # fewer than three lines

    lines = [text_line(f"l{n}", 100 + offset, 900, 200 + 50 * n) for n, offset in enumerate([0, 80, 0, 80, 80])]
    found = geometry.zigzag(lines, left=100, right=900, short=0.075, tolerance=0.005)
    assert found["mean_left"] == pytest.approx(0.06) and found["max_dev_left"] == pytest.approx(0.1)
    assert found["zigzag_left"] == 2 and found["coord_shift_left"] == 2
    assert found["std_right"] == 0 and found["zigzag_right"] == 0


def test_known_angles_give_the_parallelism_statistics() -> None:
    lines = []
    for number, angle in enumerate([0.0, 10.0, 20.0]):
        rise = 800 * math.tan(math.radians(angle))
        baseline = np.array([[100.0, 500.0], [900.0, 500.0 - rise]])  # counter-clockwise: up the page
        lines.append(TextLine(f"l{number}", box(100, 450 - rise, 900, 510), baseline, "r1"))
    found = geometry.parallelism(lines)
    assert found["angle_mean"] == pytest.approx(10) and found["angle_median"] == pytest.approx(10)
    assert found["angle_std"] == pytest.approx(10) and found["angle_max_dev"] == pytest.approx(20)
    assert found["pairwise_angle_diff_mean"] == pytest.approx(40 / 3)
    assert found["pairwise_angle_diff_median"] == pytest.approx(10)
    assert found["pairwise_angle_diff_std"] == pytest.approx(np.std([10, 20, 10], ddof=1))
    assert found["pairwise_angle_diff_max_dev"] == pytest.approx(10)


def test_graphic_and_drop_capital_areas_and_counts() -> None:
    lines = [text_line(f"l{n}", 100, 900, 200 + 50 * n) for n in range(5)]
    zones = [
        Zone("r1", "MainZone", box(100, 100, 900, 1400)),
        Zone("r2", "GraphicZone", box(100, 1000, 300, 1100)),  # 20,000
        Zone("r3", "GraphicZone", box(500, 1000, 900, 1150)),  # 60,000
        Zone("r4", "DropCapitalZone", box(100, 100, 150, 150)),  # these two overlap: one part
        Zone("r5", "DropCapitalZone", box(140, 120, 180, 160)),
    ]
    page = page_with(lines, zones)
    folio = 1000 * 1500
    assert column(page, "AreaFeature__GraphicZoneArea__graphic_num_isolated_instances") == 2
    assert column(page, "AreaFeature__GraphicZoneArea__graphic_total_area_ratio") == pytest.approx(80_000 / folio)
    assert column(page, "AreaFeature__GraphicZoneArea__graphic_area_ratio_max") == pytest.approx(60_000 / folio)
    assert column(page, "AreaFeature__GraphicZoneArea__graphic_area_ratio_std") == pytest.approx(
        np.std([20_000 / folio, 60_000 / folio], ddof=1)
    )
    assert column(page, "AreaFeature__DropCapitalArea__drop_capital_num_isolated_instances") == 1
    union = 50 * 50 + 40 * 40 - 10 * 30
    assert column(page, "AreaFeature__DropCapitalArea__drop_capital_total_area_ratio") == pytest.approx(union / folio)


def test_a_page_without_graphic_zones_counts_none() -> None:
    page = page_with([text_line(f"l{n}", 100, 900, 200 + 50 * n) for n in range(3)])
    assert column(page, "AreaFeature__GraphicZoneArea__graphic_num_isolated_instances") == 0
    assert column(page, "AreaFeature__GraphicZoneArea__graphic_area_ratio_mean") == 0


def test_line_features_need_three_lines_and_area_features_three_on_the_folio() -> None:
    family = Geometry()
    two = page_with([text_line(f"l{n}", 100, 900, 200 + 50 * n) for n in range(2)])
    assert family.measure(two) == []
    values = aggregate(family, family.measure(two))
    assert all(math.isnan(value) for value in values.values())
    three = page_with([text_line(f"l{n}", 100, 900, 200 + 50 * n) for n in range(3)])
    assert len(family.measure(three)) == 2  # the folio's row and the zone's


def test_baselines_shorter_than_seven_and_a_half_percent_of_the_zone_are_left_out() -> None:
    lines = [text_line(f"l{n}", 100, 900, 200 + 50 * n) for n in range(3)]
    short = text_line("l3", 100, 100 + 0.07 * 800, 400)
    kept = text_line("l4", 100, 100 + 0.08 * 800, 450)
    page = page_with([*lines, short])
    assert column(page, "LineFeature__LineLength__min") == pytest.approx(1.0)
    page = page_with([*lines, kept])
    assert column(page, "LineFeature__LineLength__min") == pytest.approx(0.08)
    # Two long lines and a short one: too few once it's left out, so no line features.
    page = page_with([*lines[:2], short], zones=[Zone("r1", "MainZone", box(100, 100, 900, 1400))])
    settings = GeometrySettings(fallback_zone=False)
    assert math.isnan(aggregate(Geometry(settings), Geometry(settings).measure(page))["LineFeature__LineLength__mean"])


def test_without_a_text_zone_the_folios_lines_are_measured_together() -> None:
    lines = [text_line(f"l{n}", 100, 500 + 100 * n, 200 + 50 * n, region=None) for n in range(3)]
    page = page_with(lines, zones=[])
    # The zone around them runs from 100 to 700, the longest line's end.
    assert column(page, "LineFeature__LineLength__max") == pytest.approx(1.0)
    assert column(page, "LineFeature__LineLength__min") == pytest.approx(400 / 600)
    no_fallback = Geometry(GeometrySettings(fallback_zone=False))
    assert math.isnan(aggregate(no_fallback, no_fallback.measure(page))["LineFeature__LineLength__max"])


def test_the_additive_zones_are_summed_and_the_rest_take_the_median() -> None:
    family = Geometry()
    count = "AreaFeature__GraphicZoneArea__graphic_num_isolated_instances"
    length = "LineFeature__LineLength__mean"
    assert family.aggregation(count) == "sum" and family.aggregation(length) == "median"
    values = aggregate(family, [{count: 1.0, length: 0.5}, {count: 2.0, length: 0.7}, {length: 0.9}])
    assert values[count] == 3 and values[length] == pytest.approx(0.7)


def test_text_zones_are_the_settings_and_zones_off_the_folio_are_left_out() -> None:
    lines = [text_line(f"l{n}", 100, 900, 200 + 50 * n, region="r2") for n in range(3)]
    margin = [Zone("r2", "MarginTextZone", box(100, 100, 900, 1400))]
    assert not math.isnan(column(page_with(lines, margin), "LineFeature__LineLength__mean"))
    main_only = Geometry(GeometrySettings(text_zones=("MainZone",), fallback_zone=False))
    found = aggregate(main_only, main_only.measure(page_with(lines, margin)))
    assert math.isnan(found["LineFeature__LineLength__mean"])
    off = page_with(lines, [Zone("r9", "GraphicZone", box(900, 100, 1300, 300))])
    assert column(off, "AreaFeature__GraphicZoneArea__graphic_num_isolated_instances") == 0


def test_the_family_has_the_articles_83_columns_in_nine_groups() -> None:
    family = Geometry()
    assert len(family.columns()) == 83 and len(family.groups()) == 9
    assert [len(group.flavours) for group in family.groups()] == [2, 2, 8, 7, 7, 7, 24, 18, 8]
    assert "AreaFeature__LineSimilarity__dice_median" in family.columns()
